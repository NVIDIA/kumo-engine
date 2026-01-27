#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub predict: PredictExpr,
    pub for_each: EntityRef,
    pub assuming: Option<PredicateExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictExpr {
    pub aggregation: AggregationExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregationExpr {
    pub func: AggFunc,
    pub target: AggTarget,
    pub window: Option<WindowSpec>,
    pub filter: Option<Box<PredicateExpr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggTarget {
    Column(ColumnRef),
    Star(TableRef),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggFunc {
    Sum,
    Count,
    Avg,
    Min,
    Max,
    ListDistinct,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    pub start_offset: i64,
    pub end_offset: i64,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityRef {
    pub table: String,
    pub column: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableRef {
    pub table: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnRef {
    pub table: String,
    pub column: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredicateExpr {
    Comparison {
        left: Box<Expr>,
        op: CompareOp,
        right: Box<Expr>,
    },
    And(Vec<PredicateExpr>),
    Or(Vec<PredicateExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(ColumnRef),
    Agg(Box<AggregationExpr>),
    Number(i64),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
}
