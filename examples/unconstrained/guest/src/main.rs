#![no_main]

use k256::FieldElement;

sp1_zkvm::entrypoint!(main);

pub fn main() {
    const MODULUS: &str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
    const NQR: FieldElement = FieldElement::from_u64(3);
    let value = FieldElement::from_u64(1);
    let (status, result) =
        call_sqrt_hook(value.to_bytes().as_slice(), MODULUS, NQR.to_bytes().as_slice());
    println!("status: {:?}", status);
    sp1_zkvm::io::commit(&status);
}

fn call_sqrt_hook(x: &[u8], modulus: &'static str, nqr: &[u8]) -> (u8, Vec<u8>) {
    sp1_lib::unconstrained! {
        let mut buf = Vec::new();
        buf.extend_from_slice(&32_u32.to_be_bytes());
        buf.extend_from_slice(x);
        buf.extend_from_slice(&hex::decode(modulus).unwrap());
        buf.extend_from_slice(nqr);

        sp1_lib::io::write(
            sp1_lib::io::FD_FP_SQRT,
            buf.as_slice()
        );
    }

    let status: u8 =
        sp1_lib::io::read_vec().first().copied().expect("sqrt hook should have a status");
    let result = sp1_lib::io::read_vec();

    (status, result)
}
