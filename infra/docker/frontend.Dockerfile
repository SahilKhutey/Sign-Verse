# Frontend Dockerfile for SignVerse Dashboard
# Multi-stage build for a lean production image
FROM node:18-alpine as builder

WORKDIR /app

# Install dependencies separately for caching
COPY package*.json ./
RUN npm ci

# Copy source code and build the Next.js application
COPY . .
RUN npm run build

# Production stage - Serving static Next.js assets
FROM node:18-alpine as production

WORKDIR /app

# Install 'serve' globally for high-performance static hosting
RUN npm install -g serve && \
    npm cache clean --force

# Copy only the necessary build artifacts from the builder stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./

# Create a non-root system user for security compliance
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001 && \
    chown -R nextjs:nodejs /app

USER nextjs

# Standard Next.js port
EXPOSE 3000

# Health check - Verifies the serving layer is active
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000', (res) => res.statusCode === 200 ? process.exit(0) : process.exit(1))"

# Start the application using 'serve' in single-page mode
CMD ["serve", "-s", ".next", "-p", "3000"]
