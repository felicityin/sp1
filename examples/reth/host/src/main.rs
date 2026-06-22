use sp1_sdk::{utils, ProverClient, SP1Stdin};
use sp1_sdk::Prover;

pub const RETH_ELF: &[u8] = include_bytes!("../../guest/reth");
pub const RETH_STDIN: &[u8] = include_bytes!("../../guest/stdin-24438200");

#[tokio::main]
async fn main() {
    // Setup a tracer for logging.
    utils::setup_logger();

    let stdin: SP1Stdin = bincode::deserialize(RETH_STDIN).unwrap();

    let client = ProverClient::from_env().await;
    let (_, report) = client.execute(sp1_sdk::Elf::Static(RETH_ELF), stdin).await.unwrap();
    println!("executed program with {} cycles", report.total_instruction_count());

    println!("successfully!");
}
