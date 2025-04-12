
import { cn } from "@/lib/utils";
import AmiAvatar from "@/components/AmiAvatar";
import { Message } from "@/types/chat";
import { motion } from "framer-motion";

interface MessageItemProps {
  message: Message;
  amiMood: "happy" | "neutral" | "thinking" | "sad" | "excited";
}

const MessageItem = ({ message, amiMood }: MessageItemProps) => {
  return (
    <motion.div
      className={cn(
        "flex",
        message.type === "user" ? "justify-end" : "justify-start"
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {message.type === "ami" && (
        <AmiAvatar size="sm" mood={amiMood} className="mr-2 hidden md:block" />
      )}
      
      <div
        className={cn(
          "max-w-[80%] md:max-w-[70%] rounded-3xl p-4",
          message.type === "user"
            ? "bg-primary text-white rounded-tr-none"
            : "bg-muted rounded-tl-none"
        )}
      >
        <p>{message.text}</p>
        <p className="text-xs mt-1 opacity-70">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>
      
      {message.type === "user" && (
        <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center ml-2 hidden md:flex">
          <span className="text-sm">Bạn</span>
        </div>
      )}
    </motion.div>
  );
};

export default MessageItem;
