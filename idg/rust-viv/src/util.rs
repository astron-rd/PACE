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
