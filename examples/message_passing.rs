use std::sync::{Arc, Mutex};
use std::time::Duration;

use nse;
use nse::core::{Exit, Filter, Message, System, Text};
use nse::NSE;

#[derive(Debug)]
struct Sys1 {
    counter: u64,
    // how often noop should be executed
    send: bool,
}

impl Sys1 {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Sys1 {
            counter: 0,
            send: true,
        }))
    }
}

impl System for Sys1 {
    fn consume_messages(&mut self, messages: &Vec<Message>) {
        for message in messages {
            if message.is_type::<Text>() {
                self.send = true;

                println!("Sys1 Recv - {:?}", message);
            }
        }
    }

    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>, _: Duration) {}

    fn get_messages(&mut self) -> Vec<Message> {
        let mut ret = vec![];

        if self.send {
            self.send = false;
            let msg = Message::new(Text {
                text: format!("Sys1 {}", self.counter),
            });

            println!("Sys1 Send - {:?}", msg);
            ret.push(msg);
            self.counter += 1;
        }

        if self.counter == 10 {
            ret.push(Message::new(Exit {}));
        }

        ret
    }
}

#[derive(Debug)]
struct Sys2 {
    counter: u64,
    // how often noop should be executed
    send: bool,
}

impl Sys2 {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Sys2 {
            counter: 0,
            send: false,
        }))
    }
}

impl System for Sys2 {
    fn consume_messages(&mut self, messages: &Vec<Message>) {
        for message in messages {
            if message.is_type::<Text>() {
                self.send = true;

                println!("Sys2 Recv - {:?}", message);
            }
        }
    }

    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>, _: Duration) {}

    fn get_messages(&mut self) -> Vec<Message> {
        let mut ret = vec![];

        if self.send {
            self.send = false;
            let msg = Message::new(Text {
                text: format!("Sys2 {}", self.counter),
            });

            println!("Sys2 Send - {:?}", msg);
            ret.push(msg);
            self.counter += 1;
        }

        if self.counter == 10 {
            ret.push(Message::new(Exit {}));
        }

        ret
    }
}

fn main() {
    let mut engine: NSE = NSE::new();

    let sys1 = Sys1::new();
    let sys2 = Sys2::new();

    engine.add_system(&sys1);
    engine.add_system(&sys2);

    engine.run();
}
