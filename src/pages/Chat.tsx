
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { motion, AnimatePresence } from "framer-motion";
import AmiAvatar from "@/components/AmiAvatar";
import { cn } from "@/lib/utils";
import { useChat } from "@/hooks/useChat";

// Chat components
import ChatHeader from "@/components/chat/ChatHeader";
import MobileHeader from "@/components/chat/MobileHeader";
import ChatMessages from "@/components/chat/ChatMessages";
import SuggestedQuestions from "@/components/chat/SuggestedQuestions";
import ChatInput from "@/components/chat/ChatInput";

const Chat = () => {
  const isMobile = useIsMobile();
  const [isChatExpanded, setIsChatExpanded] = useState(true); // Always expanded
  
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
    setupInactivityTimer();
  }, [setupInactivityTimer]);

  // Desktop view - Full width chat
  if (!isMobile) {
    return (
      <div className="h-[calc(100vh-theme(spacing.16))] md:h-[calc(100vh-theme(spacing.12))] flex">
        {/* Main content area */}
        {/* <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-thrive-lavender/10 to-thrive-blue/10">
          <div className="text-center space-y-6 max-w-md mx-auto">
            <AmiAvatar 
              size="xl" 
              mood={amiMood} 
              withGlow={true} 
              speakingAnimation={isSpeaking} 
              onClick={toggleChat}
            />
          </div>
        </div> */}
        
        {/* Chat panel - desktop */}
        <AnimatePresence>
          {isChatExpanded && (
            <motion.div 
              className=" border-l flex flex-col bg-background shadow-lg"
              initial={{ x: 400 }}
              animate={{ x: 0 }}
              exit={{ x: 400 }}
              transition={{ type: 'spring', damping: 30 }}
            >
              <ChatHeader 
                amiMood={amiMood} 
                isSpeaking={isSpeaking} 
                onClose={() => setIsChatExpanded(true)}
              />
              
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
          )}
        </AnimatePresence>
        
        {/* No need for floating button here anymore, it's in the Layout component */}
      </div>
    );
  }

  // Mobile view remains unchanged
  return (
    <div className="h-[calc(100vh-theme(spacing.16))] flex flex-col">
      <MobileHeader amiMood={amiMood} isSpeaking={isSpeaking} />

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
      
      {/* No need for mobile chat icon anymore, it's in the Layout component */}
    </div>
  );
};

export default Chat;
