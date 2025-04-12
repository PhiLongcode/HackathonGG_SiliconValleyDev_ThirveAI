import { useState, useEffect, useRef } from "react";
import { motion, useAnimation, useDragControls } from "framer-motion";
import { MessageCircleX, ChevronDown, ArrowUp } from "lucide-react";
import AmiAvatar from "@/components/AmiAvatar";
import { useChat } from "@/hooks/useChat";
import ChatMessages from "./ChatMessages";
import ChatInput from "./ChatInput";
import SuggestedQuestions from "./SuggestedQuestions";
import { cn } from "@/lib/utils";
import { toast } from "@/components/ui/use-toast";

const ChatBubble = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isHidden, setIsHidden] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const controls = useDragControls();
  const hideZoneRef = useRef<HTMLDivElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);
  
  const {
    messages,
    inputText,
    setInputText,
    amiMood,
    isRecording,
    isTyping,
    isSpeaking,
    setIsSpeaking,
    suggestedQuestions,
    messagesEndRef,
    inputRef,
    handleSendMessage,
    handleToggleRecording,
    handleSuggestedQuestion,
    setupInactivityTimer,
  } = useChat();

  useEffect(() => {
    const savedPosition = localStorage.getItem("amiBubblePosition");
    if (savedPosition) {
      setPosition(JSON.parse(savedPosition));
    } else {
      setPosition({ 
        x: window.innerWidth - 80, 
        y: window.innerHeight - 100 
      });
    }
    
    const amiHidden = localStorage.getItem("amiHidden") === "true";
    setIsHidden(amiHidden);
  }, []);

  useEffect(() => {
    if (position.x !== 0 && position.y !== 0) {
      localStorage.setItem("amiBubblePosition", JSON.stringify(position));
    }
  }, [position]);

  useEffect(() => {
    setupInactivityTimer();
  }, [isOpen, setupInactivityTimer]);

  const handleDragEnd = (event: any, info: any) => {
    if (!hideZoneRef.current || !bubbleRef.current) return;
    
    const hideZoneRect = hideZoneRef.current.getBoundingClientRect();
    const bubbleRect = bubbleRef.current.getBoundingClientRect();
    
    if (
      bubbleRect.bottom > hideZoneRect.top &&
      bubbleRect.left < hideZoneRect.right &&
      bubbleRect.right > hideZoneRect.left
    ) {
      setIsHidden(true);
      localStorage.setItem("amiHidden", "true");
      toast({
        title: "Ami đã ẩn",
        description: "Vuốt từ dưới lên để hiện lại Ami hoặc bấm vào góc dưới bên phải",
        duration: 3000,
      });
    } else {
      setPosition({ 
        x: info.point.x - bubbleRect.width / 2, 
        y: info.point.y - bubbleRect.height / 2 
      });
    }
    
    setIsDragging(false);
  };

  const toggleChat = () => {
    if (!isDragging) {
      setIsOpen(!isOpen);
    }
  };

  const restoreAmi = () => {
    setIsHidden(false);
    localStorage.setItem("amiHidden", "false");
  };

  useEffect(() => {
    let startY = 0;
    
    const handleTouchStart = (e: TouchEvent) => {
      startY = e.touches[0].clientY;
    };
    
    const handleTouchMove = (e: TouchEvent) => {
      if (isHidden && startY - e.touches[0].clientY > 100) {
        if (startY > window.innerHeight - 50) {
          restoreAmi();
        }
      }
    };
    
    window.addEventListener('touchstart', handleTouchStart);
    window.addEventListener('touchmove', handleTouchMove);
    
    return () => {
      window.removeEventListener('touchstart', handleTouchStart);
      window.removeEventListener('touchmove', handleTouchMove);
    };
  }, [isHidden]);

  const RestoreButton = () => (
    <motion.div 
      className="fixed bottom-4 right-4 z-50"
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      exit={{ scale: 0 }}
    >
      <button 
        onClick={restoreAmi}
        className="bg-primary text-white rounded-full p-2 shadow-md"
      >
        <MessageCircleX size={20} />
      </button>
    </motion.div>
  );

  return (
    <>
      <div 
        ref={hideZoneRef}
        className="fixed bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-gray-200/50 to-transparent pointer-events-none z-40"
      />
      
      {isHidden && <RestoreButton />}
      
      {!isHidden && (
        <motion.div
          ref={bubbleRef}
          className="fixed z-50"
          drag
          dragControls={controls}
          dragMomentum={false}
          dragTransition={{ bounceStiffness: 300, bounceDamping: 10 }}
          onDragStart={() => setIsDragging(true)}
          onDragEnd={handleDragEnd}
          initial={{ x: position.x, y: position.y }}
          animate={{ 
            x: position.x, 
            y: position.y,
            scale: isOpen ? 0.8 : 1 
          }}
          transition={{ 
            type: 'spring', 
            damping: 20, 
            stiffness: 300 
          }}
          style={{ touchAction: "none" }}
        >
          <div 
            onClick={toggleChat}
            className="bg-white rounded-full shadow-lg cursor-pointer"
          >
            <AmiAvatar 
              size="sm" 
              mood={amiMood} 
              animated={true} 
              withGlow={!isOpen}
              speakingAnimation={isSpeaking && isOpen}
            />
          </div>
        </motion.div>
      )}
      
      <motion.div 
        className={cn(
          "fixed bottom-20 right-4 w-[90vw] max-w-[350px] bg-background rounded-lg shadow-xl z-40 flex flex-col",
          "border overflow-hidden"
        )}
        initial={{ opacity: 0, scale: 0.8, y: 20 }}
        animate={{ 
          opacity: isOpen ? 1 : 0, 
          scale: isOpen ? 1 : 0.8,
          y: isOpen ? 0 : 20,
          pointerEvents: isOpen ? "auto" : "none"
        }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        style={{ height: isOpen ? "70vh" : 0 }}
      >
        <div className="flex items-center justify-between p-3 border-b">
          <div className="flex items-center">
            <AmiAvatar 
              size="sm" 
              mood={amiMood}
              speakingAnimation={isSpeaking} 
            />
            <span className="ml-2 font-medium">Ami</span>
          </div>
          <button onClick={toggleChat}>
            <ChevronDown size={20} />
          </button>
        </div>
        
        <ChatMessages 
          messages={messages} 
          amiMood={amiMood} 
          isTyping={isTyping} 
          messagesEndRef={messagesEndRef}
        />
        
        <SuggestedQuestions
          questions={suggestedQuestions}
          onSelect={handleSuggestedQuestion}
        />
        
        <ChatInput 
          inputText={inputText}
          setInputText={setInputText}
          handleSendMessage={handleSendMessage}
          handleToggleRecording={handleToggleRecording}
          isRecording={isRecording}
          isSpeaking={isSpeaking}
          setIsSpeaking={setIsSpeaking}
          inputRef={inputRef}
        />
      </motion.div>
    </>
  );
};

export default ChatBubble;
