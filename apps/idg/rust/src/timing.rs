use std::{sync::Mutex, time::Duration};

static TIMINGS: Mutex<Vec<(String, Duration)>> = Mutex::new(Vec::new());

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
    ($pretty_name: expr, $exp: expr) => {{
        let start = std::time::Instant::now();
        let result = $exp;
        let duration = start.elapsed();
        crate::timing::add_timing((
            String::from($pretty_name).to_lowercase().replace(" ", "_"),
            duration,
        ));
        println!("{:<38} {:>10.5}s", $pretty_name, duration.as_secs_f32());
        result
    }};

    ($pretty_name: expr, $name: expr, $exp: expr) => {{
        let start = std::time::Instant::now();
        let result = $exp;
        let duration = start.elapsed();
        crate::timing::add_timing((String::from($name), duration));
        println!("{:<38} {:>10.5}s", $pretty_name, duration.as_secs_f32());
        result
    }};
}

pub(crate) use time_function;

pub fn add_timing(timing: (String, Duration)) {
    TIMINGS.lock().expect("should lock").push(timing);
}

pub fn get_timings() -> Vec<(String, Duration)> {
    TIMINGS.lock().expect("should lock").clone()
}
