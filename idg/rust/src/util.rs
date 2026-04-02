macro_rules! print_header {
    ($msg: expr) => {
        println!();
        println!("{}", "=".repeat(50));
        println!("{}", $msg);
        println!("{}", "=".repeat(50));
    };
}

pub(crate) use print_header;

macro_rules! print_param {
    ($key: expr, $val: expr) => {
        println!("{:<39} {:>10}", $key, $val)
    };
}

pub(crate) use print_param;

macro_rules! time_function {
    ($name: expr, $exp: expr) => {{
        let start = std::time::Instant::now();
        let result = $exp;
        let duration = start.elapsed();
        println!("{:<37} {:>10}ms", $name, duration.as_millis());
        result
    }};
}

pub(crate) use time_function;
