use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar_inline = r#"
WHITESPACE = _{ " " | "\t" | "\r" | "\n" }

KW_PREDICT = { "PREDICT" | "predict" }
KW_FOR = { "FOR" | "for" }
KW_EACH = { "EACH" | "each" }
KW_ASSUME = { "ASSUMING" | "assuming" }
KW_WHERE = { "WHERE" | "where" }
KW_AND = { "AND" | "and" }
KW_OR = { "OR" | "or" }

ident = @{ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_" )* }
number = @{ "-"? ~ (ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT*)? | "." ~ ASCII_DIGIT+) }
string = @{ "'" ~ ( (!"'" ~ ANY)* ) ~ "'" }

query = { KW_PREDICT ~ predict_target ~ KW_FOR ~ KW_EACH ~ entity_ref ~ assuming? ~ EOI }
assuming = { (KW_ASSUME | KW_WHERE) ~ predicate }

predict_target = { cmp_expr | agg_call | column_ref }
agg_call = { agg_func ~ "(" ~ agg_args ~ ")" }
agg_func = { "SUM" | "sum" | "COUNT" | "count" | "AVG" | "avg" | "MIN" | "min" | "MAX" | "max" | "LIST_DISTINCT" | "list_distinct" }

agg_args = { agg_target ~ where_clause? ~ window_args? }
agg_target = { star_ref | column_ref }
star_ref = { ident ~ "." ~ "*" }
column_ref = { ident ~ "." ~ ident }
entity_ref = { column_ref }

where_clause = { KW_WHERE ~ predicate }
window_args = { "," ~ number ~ "," ~ number ~ "," ~ time_unit }

predicate = { or_expr }
or_expr = { and_expr ~ (KW_OR ~ and_expr)* }
and_expr = { cmp_expr ~ (KW_AND ~ cmp_expr)* }
cmp_expr = { expr ~ compare_op ~ expr }
compare_op = { "==" | "=" | "!=" | ">=" | "<=" | ">" | "<" }

expr = { agg_call | column_ref | number | string }

time_unit = { "seconds" | "second" | "minutes" | "minute" | "hours" | "hour" | "days" | "day" | "weeks" | "week" | "months" | "month" | "years" | "year" }
"#]
struct PqueryParser;

pub fn parse_query(input: &str) -> anyhow::Result<Query> {
    let mut pairs = PqueryParser::parse(Rule::query, input)?;
    let pair = pairs.next().ok_or_else(|| anyhow::anyhow!("empty parse"))?;
    build_query(pair)
}

fn build_query(pair: Pair<Rule>) -> anyhow::Result<Query> {
    let mut inner = pair.into_inner();
    consume_keyword(&mut inner, Rule::KW_PREDICT)?;
    let target_pair = inner.next().ok_or_else(|| anyhow::anyhow!("missing predict"))?;
    let predict = build_predict_target(target_pair)?;
    consume_keyword(&mut inner, Rule::KW_FOR)?;
    consume_keyword(&mut inner, Rule::KW_EACH)?;
    let entity = build_entity_ref(inner.next().ok_or_else(|| anyhow::anyhow!("missing entity"))?)?;

    let assuming = if let Some(next) = inner.next() {
        if next.as_rule() == Rule::assuming {
            Some(build_assuming(next)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(Query {
        predict,
        for_each: entity,
        assuming,
    })
}

fn build_predict_target(pair: Pair<Rule>) -> anyhow::Result<PredictExpr> {
    match pair.as_rule() {
        Rule::predict_target => {
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing predict target"))?;
            build_predict_target(inner)
        }
        Rule::agg_call => Ok(PredictExpr::Aggregation(build_agg_call(pair)?)),
        Rule::column_ref => Ok(PredictExpr::Column(build_column_ref(pair)?)),
        Rule::cmp_expr => Ok(PredictExpr::Comparison(build_cmp(pair)?)),
        _ => Err(anyhow::anyhow!("invalid predict target")),
    }
}

fn consume_keyword(pairs: &mut Pairs<Rule>, rule: Rule) -> anyhow::Result<()> {
    let next = pairs.next().ok_or_else(|| anyhow::anyhow!("missing keyword"))?;
    if next.as_rule() != rule {
        return Err(anyhow::anyhow!("expected {:?}", rule));
    }
    Ok(())
}

fn build_assuming(pair: Pair<Rule>) -> anyhow::Result<PredicateExpr> {
    let mut inner = pair.into_inner();
    if let Some(next) = inner.next() {
        match next.as_rule() {
            Rule::KW_ASSUME | Rule::KW_WHERE => {}
            _ => return Err(anyhow::anyhow!("missing assume/where keyword")),
        }
    } else {
        return Err(anyhow::anyhow!("missing assume/where keyword"));
    }
    let pred = inner.next().ok_or_else(|| anyhow::anyhow!("missing predicate"))?;
    build_predicate(pred)
}

fn build_agg_call(pair: Pair<Rule>) -> anyhow::Result<AggregationExpr> {
    let mut inner = pair.into_inner();
    let func_pair = inner.next().ok_or_else(|| anyhow::anyhow!("missing func"))?;
    let func = parse_func(func_pair.as_str())?;
    let args_pair = inner.next().ok_or_else(|| anyhow::anyhow!("missing args"))?;
    build_agg_args(func, args_pair)
}

fn build_agg_args(func: AggFunc, pair: Pair<Rule>) -> anyhow::Result<AggregationExpr> {
    let mut inner = pair.into_inner();
    let target_pair = inner.next().ok_or_else(|| anyhow::anyhow!("missing target"))?;
    let target = build_agg_target(target_pair)?;

    let mut filter: Option<Box<PredicateExpr>> = None;
    let mut window: Option<WindowSpec> = None;

    for p in inner {
        match p.as_rule() {
            Rule::where_clause => {
                let mut w_inner = p.into_inner();
                consume_keyword(&mut w_inner, Rule::KW_WHERE)?;
                let pred = w_inner.next().ok_or_else(|| anyhow::anyhow!("missing where predicate"))?;
                filter = Some(Box::new(build_predicate(pred)?));
            }
            Rule::window_args => {
                window = Some(build_window(p)?);
            }
            _ => {}
        }
    }

    Ok(AggregationExpr {
        func,
        target,
        window,
        filter,
    })
}

fn build_window(pair: Pair<Rule>) -> anyhow::Result<WindowSpec> {
    let mut inner = pair.into_inner();
    let start = inner.next().ok_or_else(|| anyhow::anyhow!("missing start"))?.as_str().parse::<i64>()?;
    let end = inner.next().ok_or_else(|| anyhow::anyhow!("missing end"))?.as_str().parse::<i64>()?;
    let unit = inner.next().ok_or_else(|| anyhow::anyhow!("missing unit"))?;
    Ok(WindowSpec {
        start_offset: start,
        end_offset: end,
        unit: parse_time_unit(unit.as_str())?,
    })
}

fn build_agg_target(pair: Pair<Rule>) -> anyhow::Result<AggTarget> {
    match pair.as_rule() {
        Rule::agg_target => {
            let mut inner = pair.into_inner();
            let next = inner.next().ok_or_else(|| anyhow::anyhow!("missing agg target"))?;
            build_agg_target(next)
        }
        Rule::star_ref => {
            let mut inner = pair.into_inner();
            let table = inner.next().ok_or_else(|| anyhow::anyhow!("missing table"))?.as_str().to_string();
            Ok(AggTarget::Star(TableRef { table }))
        }
        Rule::column_ref => {
            let col = build_column_ref(pair)?;
            Ok(AggTarget::Column(col))
        }
        _ => Err(anyhow::anyhow!("invalid agg target")),
    }
}

fn build_entity_ref(pair: Pair<Rule>) -> anyhow::Result<EntityRef> {
    match pair.as_rule() {
        Rule::entity_ref => {
            let mut inner = pair.into_inner();
            let next = inner.next().ok_or_else(|| anyhow::anyhow!("missing entity ref"))?;
            build_entity_ref(next)
        }
        Rule::column_ref => {
            let mut inner = pair.into_inner();
            let table = inner.next().ok_or_else(|| anyhow::anyhow!("missing table"))?.as_str().to_string();
            let column = inner.next().ok_or_else(|| anyhow::anyhow!("missing column"))?.as_str().to_string();
            Ok(EntityRef { table, column })
        }
        _ => Err(anyhow::anyhow!("invalid entity ref")),
    }
}

fn build_column_ref(pair: Pair<Rule>) -> anyhow::Result<ColumnRef> {
    let mut inner = pair.into_inner();
    let table = inner.next().ok_or_else(|| anyhow::anyhow!("missing table"))?.as_str().to_string();
    let column = inner.next().ok_or_else(|| anyhow::anyhow!("missing column"))?.as_str().to_string();
    Ok(ColumnRef { table, column })
}

fn build_predicate(pair: Pair<Rule>) -> anyhow::Result<PredicateExpr> {
    match pair.as_rule() {
        Rule::predicate => {
            let mut inner = pair.into_inner();
            let expr = inner.next().ok_or_else(|| anyhow::anyhow!("missing predicate expr"))?;
            build_predicate(expr)
        }
        Rule::or_expr => {
            let items = build_logic_list(pair, Rule::KW_OR)?;
            if items.len() == 1 {
                Ok(items.into_iter().next().unwrap())
            } else {
                Ok(PredicateExpr::Or(items))
            }
        }
        Rule::and_expr => {
            let items = build_logic_list(pair, Rule::KW_AND)?;
            if items.len() == 1 {
                Ok(items.into_iter().next().unwrap())
            } else {
                Ok(PredicateExpr::And(items))
            }
        }
        Rule::cmp_expr => build_cmp(pair),
        _ => Err(anyhow::anyhow!("unsupported predicate")),
    }
}

fn build_logic_list(pair: Pair<Rule>, op_rule: Rule) -> anyhow::Result<Vec<PredicateExpr>> {
    let mut items = Vec::new();
    let mut inner = pair.into_inner();
    while let Some(p) = inner.next() {
        if p.as_rule() == op_rule {
            continue;
        }
        items.push(build_predicate(p)?);
    }
    Ok(items)
}

fn build_cmp(pair: Pair<Rule>) -> anyhow::Result<PredicateExpr> {
    let mut inner = pair.into_inner();
    let left = build_expr(inner.next().ok_or_else(|| anyhow::anyhow!("missing left"))?)?;
    let op_pair = inner.next().ok_or_else(|| anyhow::anyhow!("missing op"))?;
    let right = build_expr(inner.next().ok_or_else(|| anyhow::anyhow!("missing right"))?)?;

    Ok(PredicateExpr::Comparison {
        left: Box::new(left),
        op: parse_compare_op(op_pair.as_str())?,
        right: Box::new(right),
    })
}

fn build_expr(pair: Pair<Rule>) -> anyhow::Result<Expr> {
    match pair.as_rule() {
        Rule::column_ref => Ok(Expr::Column(build_column_ref(pair)?)),
        Rule::number => Ok(Expr::Number(pair.as_str().parse::<f64>()?)),
        Rule::string => Ok(Expr::String(strip_quotes(pair.as_str()))),
        Rule::agg_call => Ok(Expr::Agg(Box::new(build_agg_call(pair)?))),
        Rule::expr => {
            let mut inner = pair.into_inner();
            let inner_pair = inner.next().ok_or_else(|| anyhow::anyhow!("missing expr inner"))?;
            build_expr(inner_pair)
        }
        _ => Err(anyhow::anyhow!("unsupported expr")),
    }
}

fn strip_quotes(input: &str) -> String {
    input.trim_start_matches('\'')
        .trim_end_matches('\'')
        .to_string()
}

fn parse_func(input: &str) -> anyhow::Result<AggFunc> {
    match input.to_ascii_uppercase().as_str() {
        "SUM" => Ok(AggFunc::Sum),
        "COUNT" => Ok(AggFunc::Count),
        "AVG" => Ok(AggFunc::Avg),
        "MIN" => Ok(AggFunc::Min),
        "MAX" => Ok(AggFunc::Max),
        "LIST_DISTINCT" => Ok(AggFunc::ListDistinct),
        _ => Err(anyhow::anyhow!("unknown func {input}")),
    }
}

fn parse_time_unit(input: &str) -> anyhow::Result<TimeUnit> {
    match input.to_ascii_lowercase().as_str() {
        "second" | "seconds" => Ok(TimeUnit::Seconds),
        "minute" | "minutes" => Ok(TimeUnit::Minutes),
        "hour" | "hours" => Ok(TimeUnit::Hours),
        "day" | "days" => Ok(TimeUnit::Days),
        "week" | "weeks" => Ok(TimeUnit::Weeks),
        "month" | "months" => Ok(TimeUnit::Months),
        "year" | "years" => Ok(TimeUnit::Years),
        _ => Err(anyhow::anyhow!("unknown time unit {input}")),
    }
}

fn parse_compare_op(input: &str) -> anyhow::Result<CompareOp> {
    match input {
        "=" | "==" => Ok(CompareOp::Eq),
        "!=" => Ok(CompareOp::Neq),
        ">" => Ok(CompareOp::Gt),
        ">=" => Ok(CompareOp::Gte),
        "<" => Ok(CompareOp::Lt),
        "<=" => Ok(CompareOp::Lte),
        _ => Err(anyhow::anyhow!("unknown compare op {input}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_list_distinct() {
        let q = "PREDICT LIST_DISTINCT(transactions.article_id) FOR EACH customers.customer_id";
        let parsed = parse_query(q).expect("parse");
        assert_eq!(parsed.for_each.table, "customers");
        match parsed.predict {
            PredictExpr::Aggregation(agg) => match agg.func {
                AggFunc::ListDistinct => {}
                _ => panic!("expected list_distinct"),
            },
            _ => panic!("expected aggregation"),
        }
    }

    #[test]
    fn parse_temporal_window() {
        let q = "PREDICT SUM(transactions.price, 0, 7, days) FOR EACH users.user_id";
        let parsed = parse_query(q).expect("parse");
        let window = match parsed.predict {
            PredictExpr::Aggregation(agg) => agg.window.expect("window"),
            _ => panic!("expected aggregation"),
        };
        assert_eq!(window.start_offset, 0);
        assert_eq!(window.end_offset, 7);
        assert_eq!(window.unit, TimeUnit::Days);
    }

    #[test]
    fn parse_assuming_predicate() {
        let q = "PREDICT SUM(transactions.price, 0, 7, days) FOR EACH user.user_id ASSUMING COUNT(promotion.* WHERE promotion.TYPE = 1, 0, 7, days) > 1";
        let parsed = parse_query(q).expect("parse");
        assert!(parsed.assuming.is_some());
    }

    #[test]
    fn parse_where_after_for_each() {
        let q = "PREDICT COUNT(transactions.*, 0, 7, days) FOR EACH customers.customer_id WHERE COUNT(transactions.*, -30, 0, days) > 0";
        let parsed = parse_query(q).expect("parse");
        assert!(parsed.assuming.is_some());
    }

    #[test]
    fn parse_decimal_literal() {
        let q = "PREDICT COUNT(transactions.* WHERE transactions.price > .1, 0, 7, days) FOR EACH customers.customer_id";
        let parsed = parse_query(q).expect("parse");
        match parsed.predict {
            PredictExpr::Aggregation(agg) => {
                let filter = agg.filter.expect("filter");
                match *filter {
                    PredicateExpr::Comparison { right, .. } => match *right {
                        Expr::Number(n) => assert!((n - 0.1).abs() < 1e-9),
                        _ => panic!("expected number"),
                    },
                    _ => panic!("expected comparison"),
                }
            }
            _ => panic!("expected aggregation"),
        }
    }

    #[test]
    fn parse_predict_column() {
        let q = "PREDICT customers.Active FOR EACH customers.customer_id";
        let parsed = parse_query(q).expect("parse");
        match parsed.predict {
            PredictExpr::Column(col) => {
                assert_eq!(col.table, "customers");
                assert_eq!(col.column, "Active");
            }
            _ => panic!("expected predict column"),
        }
    }

    #[test]
    fn parse_additional_aggs() {
        let cases = [
            ("PREDICT AVG(transactions.price) FOR EACH transactions.customer_id", AggFunc::Avg),
            ("PREDICT MIN(transactions.price) FOR EACH transactions.customer_id", AggFunc::Min),
            ("PREDICT MAX(transactions.price) FOR EACH transactions.customer_id", AggFunc::Max),
            ("PREDICT COUNT(transactions.*) FOR EACH transactions.customer_id", AggFunc::Count),
        ];
        for (q, expected) in cases {
            let parsed = parse_query(q).expect("parse");
            match parsed.predict {
                PredictExpr::Aggregation(agg) => assert_eq!(agg.func, expected),
                _ => panic!("expected aggregation"),
            }
        }
    }

    #[test]
    fn parse_where_and_or_predicate() {
        let q = "PREDICT COUNT(transactions.* WHERE transactions.article_id > 10 AND transactions.article_id <= 20, 0, 7, days) FOR EACH customers.customer_id";
        let parsed = parse_query(q).expect("parse");
        match parsed.predict {
            PredictExpr::Aggregation(agg) => assert!(agg.filter.is_some()),
            _ => panic!("expected aggregation"),
        }
    }

    #[test]
    fn parse_predict_comparison() {
        let q = "PREDICT COUNT(transactions.*, 0, 7, days) = 0 FOR EACH customers.customer_id";
        let parsed = parse_query(q).expect("parse");
        match parsed.predict {
            PredictExpr::Comparison(pred) => match pred {
                PredicateExpr::Comparison { .. } => {}
                _ => panic!("expected comparison predicate"),
            },
            _ => panic!("expected comparison predict"),
        }
    }
}
