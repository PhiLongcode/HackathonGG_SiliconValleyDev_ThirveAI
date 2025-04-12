
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

type AmiMood = "happy" | "neutral" | "thinking" | "sad" | "excited";

interface AmiAvatarProps {
  mood?: AmiMood;
  size?: "sm" | "md" | "lg" | "xl";
  animated?: boolean;
  withGlow?: boolean;
  className?: string;
  onClick?: () => void;
  speakingAnimation?: boolean;
}

const AmiAvatar = ({
  mood = "happy",
  size = "md",
  animated = true,
  withGlow = true,
  speakingAnimation = false,
  className,
  onClick,
}: AmiAvatarProps) => {
  // For a real app, you'd use actual character images
  // This is a placeholder using emoji and styling
  
  const getMoodEmoji = (mood: AmiMood) => {
    switch (mood) {
      case "happy":
        return "😊";
      case "neutral":
        return "😐";
      case "thinking":
        return "🤔";
      case "sad":
        return "😢";
      case "excited":
        return "😄";
      default:
        return "😊";
    }
  };
  
  const sizeClasses = {
    sm: "w-16 h-16 text-3xl",
    md: "w-24 h-24 text-4xl",
    lg: "w-32 h-32 text-5xl",
    xl: "w-40 h-40 text-6xl",
  };
  
  const glowSizes = {
    sm: "w-20 h-20",
    md: "w-32 h-32",
    lg: "w-40 h-40",
    xl: "w-48 h-48",
  };

  // Speaking animation effect
  const [scale, setScale] = useState(1);
  
  useEffect(() => {
    if (!speakingAnimation) return;
    
    const interval = setInterval(() => {
      setScale(prev => prev === 1 ? 1.05 : 1);
    }, 300);
    
    return () => clearInterval(interval);
  }, [speakingAnimation]);

  return (
    <div 
      className={cn(
        "ami-container relative", 
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {withGlow && (
        <div className={cn(
          "ami-glow absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 rounded-full opacity-30 blur-xl", 
          glowSizes[size]
        )}>
          <div className="absolute inset-0 bg-thrive-lavender rounded-full animate-pulse-gentle"></div>
        </div>
      )}
      <div 
        className={cn(
          "ami-avatar bg-white rounded-full flex items-center justify-center text-center shadow-lg border-4 border-thrive-lavender relative z-10",
          sizeClasses[size],
          animated && "animate-float",
          speakingAnimation && "transition-transform"
        )}
        style={{ transform: `scale(${scale})` }}
      >
        {getMoodEmoji(mood)}
      </div>

      {/* Speaking indicator */}
      {speakingAnimation && (
        <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 bg-white rounded-full px-2 py-1 shadow-md z-20">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-100"></div>
            <div className="w-2 h-2 bg-primary rounded-full animate-bounce delay-200"></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AmiAvatar;
