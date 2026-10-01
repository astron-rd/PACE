/// Measure the time to evaluate an expression and print it to stdout.
///
/// The macro evaluates to the value of the expression being measured, so you can pass it through like this:
/// ```
/// let value = time_function!("big calculation", calculate());
/// ```
///
/// A block is also an expression, so you can measure multiple statements like this:
/// ```
/// let value = time_function!("multiple calculations", {
///     let a = calculate_a();
///     let b = calculate_b();
///     b
/// });
/// ```
macro_rules! time_function {
    ($name: expr, $exp: expr) => {{
        let start = std::time::Instant::now();
        let result = $exp;
        let duration = start.elapsed();
        println!("{:<38} {:>10.5}s", $name, duration.as_secs_f32());
        result
    }};

    ($exp: expr) => {{
        let start = std::time::Instant::now();
        let result = $exp;
        let duration = start.elapsed();
        println!("{:<38} {:>10.5}s", stringify!($exp), duration.as_secs_f32());
        result
    }};
}

pub(crate) use time_function;
