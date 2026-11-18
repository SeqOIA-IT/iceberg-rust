/*!
 * Implement pruning statistics for Datafusion table
 *
 * Pruning is done on two levels:
 *
 * 1. Prune manifests based on information in manifests lists
 * 2. Prune data files based on information in manifests
 *
 * For the first level the trait [`PruningStatistics`] is implemented for the DataFusionTable. It returns the pruning information for the manifest files
 * and not the final data files.
 *
 * For the second level the trait PruningStatistics is implemented for the Manifest
*/

use std::{any::Any, sync::Arc};

use crate::error::Error as DatafusionIcebergError;
use datafusion::{
    arrow::{
        array::ArrayRef,
        datatypes::{DataType, Schema as ArrowSchema},
    },
    common::{
        tree_node::{Transformed, TreeNode},
        DataFusionError,
    },
    physical_optimizer::pruning::PruningStatistics,
    prelude::Column,
    scalar::ScalarValue,
};
use datafusion_expr::{
    expr::ScalarFunction, BinaryExpr, ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use iceberg_rust::{
    arrow::transform::transform_arrow,
    error::Error,
    spec::{
        manifest::ManifestEntry,
        manifest_list::ManifestListEntry,
        partition::{BoundPartitionField, Transform},
        schema::Schema,
        values::Value,
    },
    table::ManifestPath,
};

pub(crate) struct PruneManifests<'table, 'manifests> {
    partition_fields: &'table [BoundPartitionField<'table>],
    files: &'manifests [ManifestListEntry],
}

impl<'table, 'manifests> PruneManifests<'table, 'manifests> {
    pub(crate) fn new(
        partition_fields: &'table [BoundPartitionField<'table>],
        files: &'manifests [ManifestListEntry],
    ) -> Self {
        Self {
            partition_fields,
            files,
        }
    }
}

impl PruneManifests<'_, '_> {
    /// index et champ de partition correspondant à une colonne du prédicat (après
    /// `transform_predicate`, les colonnes portent le nom du champ de partition)
    fn partition_field(&self, column: &Column) -> Option<(usize, &BoundPartitionField<'_>)> {
        self.partition_fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.name() == column.name())
    }

    /// une valeur par manifest (None si inconnue) : les tableaux renvoyés à DataFusion doivent
    /// avoir exactement `num_containers()` éléments, dans l'ordre des manifests
    fn bounds(&self, column: &Column, lower: bool) -> Option<ArrayRef> {
        let (index, partition_field) = self.partition_field(column)?;
        let data_type = partition_field
            .field_type()
            .tranform(partition_field.transform())
            .ok()?;
        let values = self.files.iter().map(|manifest| {
            manifest
                .partitions
                .as_ref()
                .and_then(|partitions| partitions.get(index))
                .and_then(|summary| {
                    if lower {
                        summary.lower_bound.as_ref()
                    } else {
                        summary.upper_bound.as_ref()
                    }
                })
                .map(|value| value.clone().into_any())
        });
        any_iter_to_array(values, &(&data_type).try_into().ok()?).ok()
    }
}

impl PruningStatistics for PruneManifests<'_, '_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, true)
    }
    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, false)
    }
    fn num_containers(&self) -> usize {
        self.files.len()
    }
    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let (index, _) = self.partition_field(column)?;
        // la liste des manifests ne donne pas le nombre de nulls, seulement s'il y en a :
        // 0 si aucun null, inconnu sinon
        let null_counts = self.files.iter().map(|manifest| {
            manifest
                .partitions
                .as_ref()
                .and_then(|partitions| partitions.get(index))
                .and_then(|summary| (!summary.contains_null).then_some(0i64))
        });
        ScalarValue::iter_to_array(null_counts.map(ScalarValue::Int64)).ok()
    }
    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<datafusion::arrow::array::BooleanArray> {
        None
    }

    fn row_counts(&self, _column: &Column) -> Option<ArrayRef> {
        // nombre de lignes vivantes du manifest = ADDED + EXISTING (et pas seulement ADDED :
        // un manifest réécrit par overwrite/merge n'a que des entrées EXISTING, avec
        // added_rows_count = 0 ; DataFusion l'élaguait alors via `null_count != row_count`)
        ScalarValue::iter_to_array(
            self.files
                .iter()
                .map(|x| match (x.added_rows_count, x.existing_rows_count) {
                    (Some(added), Some(existing)) => Some(added + existing),
                    _ => None,
                })
                .map(ScalarValue::Int64),
        )
        .ok()
    }
}

pub(crate) struct PruneDataFiles<'table, 'manifests> {
    schema: &'table Schema,
    arrow_schema: &'table ArrowSchema,
    files: &'manifests [(ManifestPath, ManifestEntry)],
}

impl<'table, 'manifests> PruneDataFiles<'table, 'manifests> {
    pub(crate) fn new(
        schema: &'table Schema,
        arrow_schema: &'table ArrowSchema,
        files: &'manifests [(ManifestPath, ManifestEntry)],
    ) -> Self {
        Self {
            schema,
            arrow_schema,
            files,
        }
    }
}

impl PruningStatistics for PruneDataFiles<'_, '_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        let column_id = self.schema.fields().get_name(&column.name)?.id;
        let datatype = self
            .arrow_schema
            .field_with_name(&column.name)
            .ok()?
            .data_type();
        let min_values =
            self.files
                .iter()
                .map(|manifest| match &manifest.1.data_file().lower_bounds() {
                    Some(map) => map
                        .get(&{ column_id })
                        .map(|value| value.clone().into_any()),
                    None => None,
                });
        any_iter_to_array(min_values, datatype).ok()
    }
    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        let column_id = self.schema.fields().get_name(&column.name)?.id;
        let datatype = self
            .arrow_schema
            .field_with_name(&column.name)
            .ok()?
            .data_type();
        let max_values =
            self.files
                .iter()
                .map(|manifest| match &manifest.1.data_file().upper_bounds() {
                    Some(map) => map
                        .get(&{ column_id })
                        .map(|value| value.clone().into_any()),
                    None => None,
                });
        any_iter_to_array(max_values, datatype).ok()
    }
    fn num_containers(&self) -> usize {
        self.files.len()
    }
    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let column_id = self.schema.fields().get_name(&column.name)?.id;
        let null_counts =
            self.files.iter().map(
                |manifest| match &manifest.1.data_file().null_value_counts() {
                    Some(map) => map.get(&{ column_id }).copied(),
                    None => None,
                },
            );
        ScalarValue::iter_to_array(null_counts.map(ScalarValue::Int64)).ok()
    }
    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<datafusion::arrow::array::BooleanArray> {
        None
    }

    fn row_counts(&self, column: &Column) -> Option<ArrayRef> {
        let column_id = self.schema.fields().get_name(&column.name)?.id;
        let null_counts =
            self.files
                .iter()
                .map(|manifest| match &manifest.1.data_file().value_counts() {
                    Some(map) => map.get(&{ column_id }).copied(),
                    None => None,
                });
        ScalarValue::iter_to_array(null_counts.map(ScalarValue::Int64)).ok()
    }
}

fn any_iter_to_array(
    iter: impl Iterator<Item = Option<Box<dyn Any>>>,
    datatype: &DataType,
) -> Result<ArrayRef, DataFusionError> {
    match datatype {
        DataType::Boolean => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Boolean(opt.and_then(|value| Some(*value.downcast::<bool>().ok()?)))
        })),
        DataType::Int32 => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Int32(opt.and_then(|value| Some(*value.downcast::<i32>().ok()?)))
        })),
        DataType::Int64 => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Int64(opt.and_then(|value| Some(*value.downcast::<i64>().ok()?)))
        })),
        DataType::Float32 => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Float32(opt.and_then(|value| Some(*value.downcast::<f32>().ok()?)))
        })),
        DataType::Float64 => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Float64(opt.and_then(|value| Some(*value.downcast::<f64>().ok()?)))
        })),
        DataType::Date64 => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Date64(opt.and_then(|value| Some(*value.downcast::<i64>().ok()?)))
        })),
        DataType::Time64(_) => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Time64Microsecond(
                opt.and_then(|value| Some(*value.downcast::<i64>().ok()?)),
            )
        })),
        DataType::Timestamp(_, _) => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::TimestampMicrosecond(
                opt.and_then(|value| Some(*value.downcast::<i64>().ok()?)),
                None,
            )
        })),
        DataType::Utf8 => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Utf8(opt.and_then(|value| Some(*value.downcast::<String>().ok()?)))
        })),
        DataType::FixedSizeBinary(_) => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Binary(opt.and_then(|value| Some(*value.downcast::<Vec<u8>>().ok()?)))
        })),
        DataType::Binary => ScalarValue::iter_to_array(iter.map(|opt| {
            ScalarValue::Binary(opt.and_then(|value| Some(*value.downcast::<Vec<u8>>().ok()?)))
        })),
        _ => Err(DataFusionError::Internal(
            "Arrow datatype not supported for pruning.".to_string(),
        )),
    }
}

pub(crate) fn transform_predicate(
    expr: Expr,
    partition_fields: &[BoundPartitionField],
) -> Result<Expr, DataFusionError> {
    expr.transform_down(|expr| match expr {
        Expr::BinaryExpr(bin) => match (*bin.left, *bin.right) {
            (Expr::Column(column), right) => {
                let field = partition_fields
                    .iter()
                    .find(|x| x.source_name() == column.name())
                    .unwrap();
                Ok(Transformed::yes(Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(Expr::Column(Column::new(
                        column.relation,
                        field.name().to_owned(),
                    ))),
                    bin.op,
                    Box::new(transform_literal(right, field.transform())?),
                ))))
            }
            (left, Expr::Column(column)) => {
                let field = partition_fields
                    .iter()
                    .find(|x| x.source_name() == column.name())
                    .unwrap();
                Ok(Transformed::yes(Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(Expr::Column(Column::new(
                        column.relation,
                        field.name().to_owned(),
                    ))),
                    bin.op,
                    Box::new(transform_literal(left, field.transform())?),
                ))))
            }
            (left, right) => Ok(Transformed::no(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(left),
                bin.op,
                Box::new(right),
            )))),
        },
        x => Ok(Transformed::no(x)),
    })
    .map(|x| x.data)
}

fn transform_literal(expr: Expr, transform: &Transform) -> Result<Expr, DataFusionError> {
    match transform {
        Transform::Year => Ok(Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(ScalarUDF::new_from_impl(DateTransform::new())),
            vec![Expr::Literal(ScalarValue::new_utf8("year"), None), expr],
        ))),
        Transform::Month => Ok(Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(ScalarUDF::new_from_impl(DateTransform::new())),
            vec![Expr::Literal(ScalarValue::new_utf8("month"), None), expr],
        ))),
        Transform::Day => Ok(Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(ScalarUDF::new_from_impl(DateTransform::new())),
            vec![Expr::Literal(ScalarValue::new_utf8("day"), None), expr],
        ))),
        Transform::Hour => Ok(Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(ScalarUDF::new_from_impl(DateTransform::new())),
            vec![Expr::Literal(ScalarValue::new_utf8("hour"), None), expr],
        ))),
        _ => Ok(expr),
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct DateTransform {
    signature: Signature,
}

impl DateTransform {
    fn new() -> Self {
        let signature = Signature {
            type_signature: TypeSignature::OneOf(vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Date32]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Timestamp(datafusion::arrow::datatypes::TimeUnit::Microsecond, None),
                ]),
            ]),
            volatility: Volatility::Immutable,
            parameter_names: None,
        };
        Self { signature }
    }
}

impl ScalarUDFImpl for DateTransform {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "date_transform"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Int32)
    }

    fn invoke_with_args(
        &self,
        args: ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        let args = args.args;
        let transform = &args[0];
        let array = &args[1];
        let ColumnarValue::Scalar(ScalarValue::Utf8(Some(transform))) = transform else {
            return Err(DataFusionError::External(Box::new(Error::InvalidFormat(
                "Partition transform".to_owned(),
            ))));
        };
        let transform = match transform.as_str() {
            "year" => Ok(Transform::Year),
            "month" => Ok(Transform::Month),
            "day" => Ok(Transform::Day),
            "hour" => Ok(Transform::Hour),
            _ => Err(DataFusionError::External(Box::new(Error::InvalidFormat(
                "Partition transform".to_owned(),
            )))),
        }?;
        match array {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(transform_arrow(
                array.clone(),
                &transform,
            )?)),
            ColumnarValue::Scalar(scalar) => Ok(ColumnarValue::Scalar(
                value_to_scalarvalue(
                    scalarvalue_to_value(scalar)
                        .map_err(DatafusionIcebergError::from)?
                        .transform(&transform)
                        .map_err(DatafusionIcebergError::from)?,
                )
                .map_err(DatafusionIcebergError::from)?,
            )),
        }
    }
}

fn scalarvalue_to_value(scalar: &ScalarValue) -> Result<Value, Error> {
    match scalar {
        ScalarValue::Boolean(x) => Ok(Value::Boolean(x.ok_or(Error::InvalidFormat(
            "Value can't be null when converting to iceberg value".to_owned(),
        ))?)),
        ScalarValue::Int32(x) => Ok(Value::Int(x.ok_or(Error::InvalidFormat(
            "Value can't be null when converting to iceberg value".to_owned(),
        ))?)),
        ScalarValue::Int64(x) => Ok(Value::LongInt(x.ok_or(Error::InvalidFormat(
            "Value can't be null when converting to iceberg value".to_owned(),
        ))?)),
        ScalarValue::Date32(x) => Ok(Value::Date(x.ok_or(Error::InvalidFormat(
            "Value can't be null when converting to iceberg value".to_owned(),
        ))?)),
        ScalarValue::Time64Microsecond(x) => Ok(Value::Time(x.ok_or(Error::InvalidFormat(
            "Value can't be null when converting to iceberg value".to_owned(),
        ))?)),
        ScalarValue::TimestampMicrosecond(x, _) => Ok(Value::Timestamp(x.ok_or(
            Error::InvalidFormat("Value can't be null when converting to iceberg value".to_owned()),
        )?)),
        x => Err(Error::NotSupported(format!(
            "Transforming {x} to iceberg value"
        ))),
    }
}

fn value_to_scalarvalue(value: Value) -> Result<ScalarValue, Error> {
    match value {
        Value::Boolean(x) => Ok(ScalarValue::Boolean(Some(x))),
        Value::Int(x) => Ok(ScalarValue::Int32(Some(x))),
        Value::LongInt(x) => Ok(ScalarValue::Int64(Some(x))),
        Value::Date(x) => Ok(ScalarValue::Date32(Some(x))),
        Value::Time(x) => Ok(ScalarValue::Time64Microsecond(Some(x))),
        Value::Timestamp(x) => Ok(ScalarValue::TimestampMicrosecond(Some(x), None)),
        x => Err(Error::NotSupported(format!(
            "Transforming {x} to iceberg value"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema as ArrowSchema},
        common::DFSchema,
        execution::context::ExecutionProps,
        physical_expr::create_physical_expr,
        physical_optimizer::pruning::PruningPredicate,
        prelude::{col, lit},
    };
    use iceberg_rust::spec::{
        manifest_list::{Content, FieldSummary, ManifestListEntry},
        partition::{PartitionField, PartitionSpec, Transform},
        schema::Schema,
        table_metadata::{partition_fields, FormatVersion},
        types::{PrimitiveType, StructField, Type},
        values::Value,
    };

    use super::{transform_predicate, PruneManifests};

    fn manifest(
        added_rows: i64,
        existing_rows: i64,
        bounds: Option<(&str, &str)>,
    ) -> ManifestListEntry {
        ManifestListEntry {
            format_version: FormatVersion::V2,
            manifest_path: "m.avro".to_owned(),
            manifest_length: 1,
            partition_spec_id: 0,
            content: Content::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 1,
            added_files_count: Some(if added_rows > 0 { 1 } else { 0 }),
            existing_files_count: Some(if existing_rows > 0 { 1 } else { 0 }),
            deleted_files_count: Some(0),
            added_rows_count: Some(added_rows),
            existing_rows_count: Some(existing_rows),
            deleted_rows_count: Some(0),
            partitions: bounds.map(|(lo, hi)| {
                vec![FieldSummary {
                    contains_null: false,
                    contains_nan: false,
                    lower_bound: Some(Value::String(lo.to_owned())),
                    upper_bound: Some(Value::String(hi.to_owned())),
                }]
            }),
            key_metadata: None,
        }
    }

    /// un manifest réécrit (que des entrées EXISTING, added_rows_count = 0) ne doit pas être
    /// élagué, et un manifest sans résumé de partition ne doit pas décaler les suivants
    #[test]
    fn test_prune_manifests_existing_entries() {
        let schema = Schema::builder()
            .with_struct_field(StructField {
                id: 1,
                name: "rec_code".to_string(),
                required: true,
                field_type: Type::Primitive(PrimitiveType::String),
                doc: None,
            })
            .build()
            .unwrap();
        let spec = PartitionSpec::builder()
            .with_partition_field(PartitionField::new(1, 1000, "rec_code", Transform::Identity))
            .build()
            .unwrap();
        let fields = partition_fields(&spec, &schema).unwrap();
        let partition_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "rec_code",
            DataType::Utf8,
            false,
        )]));

        let manifests = vec![
            manifest(0, 100, Some(("a", "a"))), // réécrit, contient 'a'
            manifest(100, 0, Some(("a", "a"))), // ajouté, contient 'a'
            manifest(0, 100, Some(("b", "b"))), // ne contient pas 'a'
            manifest(0, 100, None),             // pas de résumé : inconnu
            manifest(10, 0, Some(("a", "a"))),  // après un manifest sans résumé
            manifest(10, 0, Some(("c", "c"))),  // ne contient pas 'a'
        ];

        let predicate = transform_predicate(col("rec_code").eq(lit("a")), &fields).unwrap();
        let df_schema = DFSchema::try_from(partition_schema.as_ref().clone()).unwrap();
        let physical =
            create_physical_expr(&predicate, &df_schema, &ExecutionProps::new()).unwrap();
        let pruning = PruningPredicate::try_new(physical, partition_schema).unwrap();
        let keep = pruning
            .prune(&PruneManifests::new(&fields, &manifests))
            .unwrap();

        assert_eq!(keep, vec![true, true, false, true, true, false]);
    }
}
