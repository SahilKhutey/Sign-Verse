# Redis Dockerfile
# Optimized for SignVerse AI caching and session management
FROM redis:7-alpine

# Set directory permissions for security
WORKDIR /data

# Copy custom configuration
# Note: Creating this file in the same directory as part of this turn
COPY redis.conf /usr/local/etc/redis/redis.conf

# Create data directory with correct ownership for the 'redis' user
# Already handled by the base image usually, but explicitly ensuring here
RUN mkdir -p /data && chown redis:redis /data

# Standard Redis port
EXPOSE 6379

# Health check - Periodically checks that Redis is responding to 'PING'
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD redis-cli ping | grep PONG || exit 1

# Start Redis with the custom configuration
# This allows us to tune memory limits and persistence policies
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
