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
