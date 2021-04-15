use nse;
use nse::NSE;
use nse::rendering::RenderSystem;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = Box::new(RenderSystem::new(&engine));

    engine.add_system(&render_system);

    engine.run();
}
