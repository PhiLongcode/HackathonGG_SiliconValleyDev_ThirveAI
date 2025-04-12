
import { useRef, useEffect } from "react";
import { Message, AmiMood } from "@/types/chat";
import { getEncouragementMessage } from "@/utils/chatUtils";

type InactivityHandler = (message: Message, mood: AmiMood) => void;

interface UseInactivityProps {
  isEnabled: boolean;
  onInactiveMessage: InactivityHandler;
}

export function useInactivity({ isEnabled, onInactiveMessage }: UseInactivityProps) {
  const lastInteractionTimeRef = useRef<number>(Date.now());
  const inactivityTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Update last interaction time when user interacts
  useEffect(() => {
    const updateLastInteractionTime = () => {
      lastInteractionTimeRef.current = Date.now();
    };

    // Listen for user interactions
    window.addEventListener("click", updateLastInteractionTime);
    window.addEventListener("keydown", updateLastInteractionTime);
    window.addEventListener("mousemove", updateLastInteractionTime);
    window.addEventListener("touchstart", updateLastInteractionTime);

    return () => {
      window.removeEventListener("click", updateLastInteractionTime);
      window.removeEventListener("keydown", updateLastInteractionTime);
      window.removeEventListener("mousemove", updateLastInteractionTime);
      window.removeEventListener("touchstart", updateLastInteractionTime);
    };
  }, []);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (inactivityTimerRef.current) {
        clearInterval(inactivityTimerRef.current);
      }
    };
  }, []);

  const setupInactivityTimer = () => {
    // Clear any existing timer
    if (inactivityTimerRef.current) {
      clearInterval(inactivityTimerRef.current);
    }

    if (!isEnabled) return;

    const checkInactivity = () => {
      const now = Date.now();
      const inactiveTime = now - lastInteractionTimeRef.current;
      
      // If inactive for 5 minutes (300000 ms) and chat is expanded
      if (inactiveTime > 300000) {
        const randomMessage = getEncouragementMessage();
        
        // Add encouragement message from Ami
        const amiMessage: Message = {
          id: Date.now().toString(),
          type: "ami",
          text: randomMessage,
          timestamp: new Date(),
        };
        
        onInactiveMessage(amiMessage, "neutral");
        
        // Reset the timer
        lastInteractionTimeRef.current = now;
      }
    };
    
    // Set up the inactivity check timer (check every minute)
    inactivityTimerRef.current = setInterval(checkInactivity, 60000);
  };

  return { setupInactivityTimer };
}
