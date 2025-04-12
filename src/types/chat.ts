
export type MessageType = "user" | "ami";
export type AmiMood = "happy" | "neutral" | "thinking" | "sad" | "excited";

export interface Message {
  id: string;
  type: MessageType;
  text: string;
  timestamp: Date;
}
