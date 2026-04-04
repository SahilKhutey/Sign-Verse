import { Html, Head, Main, NextScript } from 'next/document';

/**
 * Custom Document Structure
 * 
 * Sets the default language and global meta-tags for SEO and initial viewport.
 * Controls the overall SSR structure (Head, Body, Main, Scripts).
 */
export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta charSet="UTF-8" />
        {/* SignVerse Meta Info */}
        <meta name="description" content="SignVerse - AI-Powered Pose Estimation and Simulation Ecosystem" />
        <meta name="theme-color" content="#4f46e5" />
        
        {/* Performance / Fonts */}
        <link rel="icon" href="/favicon.ico" />
        
        {/* Anti-aliasing / Base Body Styles */}
      </Head>
      <body className="antialiased selection:bg-indigo-100 selection:text-indigo-900 overflow-x-hidden">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
