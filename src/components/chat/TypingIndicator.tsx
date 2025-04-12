
import { motion } from "framer-motion";
import AmiAvatar from "@/components/AmiAvatar";

const TypingIndicator = () => {
  return (
    <div className="flex justify-start">
      <AmiAvatar size="sm" mood="thinking" className="mr-2 hidden md:block" />
      <div className="bg-muted rounded-3xl rounded-tl-none p-4 flex items-center space-x-2">
        <motion.div 
          className="w-2 h-2 bg-muted-foreground/70 rounded-full"
          animate={{ scale: [0.8, 1.2, 0.8] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        <motion.div 
          className="w-2 h-2 bg-muted-foreground/70 rounded-full"
          animate={{ scale: [0.8, 1.2, 0.8] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
        />
        <motion.div 
          className="w-2 h-2 bg-muted-foreground/70 rounded-full"
          animate={{ scale: [0.8, 1.2, 0.8] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
        />
      </div>
    </div>
  );
};

export default TypingIndicator;
