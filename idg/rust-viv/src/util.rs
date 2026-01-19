macro_rules! print_header {
    ($msg: expr) => {
        println!("\n==== {} ====", $msg)
    };
}

pub(crate) use print_header;
