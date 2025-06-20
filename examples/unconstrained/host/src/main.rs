use sp1_sdk::{include_elf, utils, ProverClient, SP1Stdin};
pub const ELF: &[u8] = include_elf!("unconstrained");

fn main() {
    utils::setup_logger();

    let stdin = SP1Stdin::new();

    let client = ProverClient::new();
    let (public_values, report) = client.execute(ELF, &stdin).run().expect("failed to prove");

    println!("report: {}", report);
    println!("public_values: {:?}", public_values);

    let (pk, vk) = client.setup(ELF);
    let proof = client.prove(&pk, &stdin).run().unwrap();
    client.verify(&proof, &vk).expect("verification failed");
    println!("done");
}
