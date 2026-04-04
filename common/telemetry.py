import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_telemetry(app=None, service_name="signverse-service"):
    """
    Configure Distributed Tracing with OpenTelemetry.
    Exports traces to the console for demonstration; in production, 
    this would use Jaeger or OTLP exporters.
    """
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    if app:
        FastAPIInstrumentor.instrument_app(app)
    
    return trace.get_tracer(service_name)
