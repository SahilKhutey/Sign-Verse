import React from 'react';
import type { AppProps } from 'next/app';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import '@/styles/globals.css';

/**
 * Global Query Client Configuration
 * 
 * Configuring standard caching and retry policies for the dashboard.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes fresh time
      retry: 1, // Only retry once to avoid excessive API load
      refetchOnWindowFocus: false, // Prevent background refetches on tab switch
    },
  },
});

/**
 * Main App Wrapper
 * 
 * Sets up global providers including React Query and injects global styles.
 */
export default function App({ Component, pageProps }: AppProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <Component {...pageProps} />
      <ReactQueryDevtools initialIsOpen={false} position="bottom" />
    </QueryClientProvider>
  );
}
