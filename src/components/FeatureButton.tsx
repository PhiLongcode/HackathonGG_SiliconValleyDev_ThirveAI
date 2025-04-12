
import { ReactNode } from "react";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface FeatureButtonProps {
  to: string;
  icon: ReactNode;
  label: string;
  description?: string;
  className?: string;
  onClick?: () => void;
}

const FeatureButton = ({
  to,
  icon,
  label,
  description,
  className,
  onClick,
}: FeatureButtonProps) => {
  return (
    <Link
      to={to}
      className={cn(
        "thrive-card hover:scale-105 flex flex-col items-center p-6 gap-4 transform transition-all duration-300 relative overflow-hidden",
        className
      )}
      onClick={onClick}
    >
      <div className="thrive-icon-button text-2xl bg-accent text-accent-foreground relative z-10">
        {icon}
      </div>
      <div className="text-center relative z-10">
        <h3 className="font-medium text-lg text-white">{label}</h3>
        {description && (
          <p className="text-sm text-white/80 mt-1">{description}</p>
        )}
      </div>
      
      {/* Gradient overlay to ensure text is readable */}
      <div className="absolute inset-0 bg-black/10 hover:bg-black/0 transition-colors duration-300"></div>
      
      {/* Interactive animation on hover */}
      <div className="absolute top-0 left-0 w-full h-1 bg-white/20 transform -translate-x-full group-hover:translate-x-0 transition-transform duration-700"></div>
      <div className="absolute bottom-0 right-0 w-full h-1 bg-white/20 transform translate-x-full group-hover:translate-x-0 transition-transform duration-700"></div>
    </Link>
  );
};

export default FeatureButton;
