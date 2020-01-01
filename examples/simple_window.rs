use nse;
use nse::NSE;
use nse::rendering::{RenderSystem};

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = Box::new(RenderSystem::new(&engine.event_loop));

    engine.system_manager.add_system(render_system);

    engine.run();
}
