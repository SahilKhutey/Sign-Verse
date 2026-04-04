import React from 'react';
import { Layout } from '@/components/Layout/Layout';
import { VideoUpload } from '@/components/Upload/VideoUpload';
import { UploadHistory } from '@/components/Upload/UploadHistory';

/**
 * Video Upload Page
 * 
 * Provides an interface for dragging and dropping video files for analysis
 * and viewing the history of previous uploads.
 */
export default function UploadPage() {
  return (
    <Layout>
      <div className="space-y-8 animate-in fade-in duration-500">
        {/* Page Header */}
        <div className="border-b border-gray-200 pb-5 dark:border-gray-800">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
            Video Upload
          </h1>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Submit MP4, AVI, or MOV files to the AI pipeline for pose estimation and physical simulation.
          </p>
        </div>

        {/* Upload Hub */}
        <section aria-labelledby="upload-section">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 dark:bg-gray-900 dark:border-gray-800 overflow-hidden">
            <div className="p-1">
              <VideoUpload />
            </div>
          </div>
        </section>

        {/* Longitudinal Archival / History */}
        <section aria-labelledby="history-section" className="pt-4">
          <div className="flex items-center justify-between mb-4">
            <h2 id="history-section" className="text-lg font-semibold text-gray-900 dark:text-white">
              Recent Processing History
            </h2>
          </div>
          <UploadHistory />
        </section>
      </div>
    </Layout>
  );
}
