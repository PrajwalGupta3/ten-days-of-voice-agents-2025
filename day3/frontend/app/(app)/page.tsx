import { Suspense } from "react";
import { App } from "@/components/app/app";
import { APP_CONFIG_DEFAULTS } from "@/app-config";

export default function Page() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 gap-8">
      
      {/* Container with Glass Effect */}
      <div className="glass-panel p-12 flex flex-col items-center max-w-lg w-full relative overflow-hidden">
        
        {/* Background decorative blob */}
        <div className="absolute top-[-50%] left-1/2 -translate-x-1/2 w-[300px] h-[300px] bg-[#FF3E56] opacity-10 blur-[100px] rounded-full pointer-events-none"></div>

        {/* BRAND HEADER */}
        <div className="flex flex-col items-center gap-6 mb-8 relative z-10">
          <img 
            src="https://cdn-images.cure.fit/www-curefit-com/image/upload/c_fill,w_120,ar_3.86,q_auto:eco,dpr_2,f_auto,fl_progressive/dpr_2/image/test/brand-logo/vman-and-white-cult-text.png" 
            alt="Cult.Fit Logo" 
            className="h-12 object-contain drop-shadow-lg"
          />
          <div className="text-center space-y-2">
            <h1 className="text-2xl font-bold tracking-widest text-white uppercase">
              AI Wellness Coach
            </h1>
            <div className="h-1 w-12 bg-[#FF3E56] mx-auto rounded-full"></div> {/* Divider */}
            <p className="text-gray-400 text-sm font-medium tracking-wide">
              MIND • BODY • SPIRIT
            </p>
          </div>
        </div>

        {/* APP INTERFACE */}
        <div className="w-full flex justify-center relative z-10">
          <Suspense fallback={<div className="text-white/50 animate-pulse">Initializing...</div>}>
             <App appConfig={APP_CONFIG_DEFAULTS} />
          </Suspense>
        </div>
      </div>
    </main>
  );
}