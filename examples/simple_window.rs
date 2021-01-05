use nse;
use nse::rendering::RenderSystem;
use nse::NSE;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = Box::new(RenderSystem::new(&engine));

    engine.add_system(&render_system);

    engine.run();
}
