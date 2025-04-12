
import { RefObject } from "react";
import { Message } from "@/types/chat";
import MessageItem from "./MessageItem";
import TypingIndicator from "./TypingIndicator";

interface ChatMessagesProps {
  messages: Message[];
  amiMood: "happy" | "neutral" | "thinking" | "sad" | "excited";
  isTyping: boolean;
  messagesEndRef: RefObject<HTMLDivElement>;
}

const ChatMessages = ({ messages, amiMood, isTyping, messagesEndRef }: ChatMessagesProps) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message) => (
        <MessageItem 
          key={message.id} 
          message={message} 
          amiMood={amiMood} 
        />
      ))}
      
      {isTyping && <TypingIndicator />}
      
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatMessages;
