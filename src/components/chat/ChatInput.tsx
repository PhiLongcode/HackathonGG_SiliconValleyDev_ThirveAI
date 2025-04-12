
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Mic, MicOff, Send, Smile, Volume, VolumeX } from "lucide-react";
import { RefObject } from "react";

interface ChatInputProps {
  inputText: string;
  setInputText: (text: string) => void;
  handleSendMessage: () => void;
  handleToggleRecording: () => void;
  isRecording: boolean;
  isSpeaking: boolean;
  setIsSpeaking: (isSpeaking: boolean) => void;
  inputRef: RefObject<HTMLInputElement>;
}

const ChatInput = ({
  inputText,
  setInputText,
  handleSendMessage,
  handleToggleRecording,
  isRecording,
  isSpeaking,
  setIsSpeaking,
  inputRef,
}: ChatInputProps) => {
  return (
    <div className="border-t p-4 bg-background">
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="icon"
          className="rounded-full"
          onClick={() => {}}
        >
          <Smile className="h-5 w-5" />
        </Button>
        
        <Input
          ref={inputRef}
          className="flex-1 rounded-full bg-muted border-0"
          placeholder="Nhắn tin cho Ami..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
        />
        
        <Button
          variant="outline"
          size="icon"
          className={cn(
            "rounded-full",
            isRecording && "bg-red-100 text-red-500 border-red-200"
          )}
          onClick={handleToggleRecording}
        >
          {isRecording ? (
            <MicOff className="h-5 w-5" />
          ) : (
            <Mic className="h-5 w-5" />
          )}
        </Button>
        
        <Button
          variant="outline"
          size="icon"
          className={cn(
            "rounded-full",
            isSpeaking ? "bg-primary/20 text-primary" : "text-muted-foreground"
          )}
          onClick={() => setIsSpeaking(!isSpeaking)}
        >
          {isSpeaking ? (
            <Volume className="h-5 w-5" />
          ) : (
            <VolumeX className="h-5 w-5" />
          )}
        </Button>
        
        <Button
          size="icon"
          className="rounded-full"
          onClick={handleSendMessage}
          disabled={!inputText.trim()}
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

export default ChatInput;
