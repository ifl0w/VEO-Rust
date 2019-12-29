use crossbeam::crossbeam_channel::unbounded;
use std::fmt::Debug;
use std::any::TypeId;

pub trait Payload: Debug { }

#[derive(Debug, Copy, Clone)]
pub struct Message {
    pub code: TypeId,
//    pub data: *const dyn Payload
}

#[derive(Debug)]
pub struct ExitMessage {}
impl Payload for ExitMessage {}

#[derive(Debug)]
pub struct TextMessage {
    text: String
}
impl Payload for TextMessage {}

pub struct MessageManager {
    // maps message type to listener objects
    pub sender: crossbeam::crossbeam_channel::Sender<Message>,
    pub receiver: crossbeam::crossbeam_channel::Receiver<Message>
}

impl MessageManager {
    pub fn new() -> Self {
        let (send, receive) = unbounded();

        MessageManager {
            sender: send,
            receiver: receive
        }
    }
}
