//! A JSON value as the skill's messages print it: `${value}` and
//! `JSON.stringify(value)`.

use serde_json::Value;

use crate::js::js_string;

/// `${value}`: a number as JavaScript writes it, a string as it is, an
/// array's items joined by commas, an object as `[object Object]`.
pub(crate) fn js_display(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => js_string(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|i| match i {
                Value::Null => String::new(),
                other => js_display(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

/// `JSON.stringify(value)`: compact, numbers as JavaScript writes them.
pub(crate) fn js_stringify(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(f64::NAN);
            if x.is_finite() {
                js_string(x)
            } else {
                "null".to_string()
            }
        }
        Value::String(s) => Value::String(s.clone()).to_string(),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(js_stringify).collect::<Vec<_>>().join(",")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}:{}", Value::String(k.clone()), js_stringify(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

/// The number `v` holds, when it is one.
pub(crate) fn number(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// `Number.isInteger(v)`.
pub(crate) fn is_integer(v: &Value) -> bool {
    number(v).is_some_and(|x| x.is_finite() && x.fract() == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn values_print_as_javascript_prints_them() {
        assert_eq!(js_display(&json!(1.0)), "1");
        assert_eq!(js_display(&json!("x")), "x");
        assert_eq!(js_display(&json!([1, null, "a"])), "1,,a");
        assert_eq!(js_display(&json!({"a": 1})), "[object Object]");
        assert_eq!(js_stringify(&json!("x")), "\"x\"");
        assert_eq!(js_stringify(&json!([1.0, {"a": 0.5}])), "[1,{\"a\":0.5}]");
        assert!(is_integer(&json!(4.0)));
        assert!(!is_integer(&json!(4.5)));
        assert!(!is_integer(&json!("4")));
    }
}
