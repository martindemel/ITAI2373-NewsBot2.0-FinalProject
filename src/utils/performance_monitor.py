#!/usr/bin/env python3
"""
Performance Monitor for NewsBot 2.0
Production-ready performance monitoring, metrics collection, and system health tracking
"""

import time
import psutil
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import functools
import traceback
import gc
import json
import os
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: str
    component: str
    category: str
    metadata: Dict[str, Any] = None

@dataclass
class SystemHealthSnapshot:
    """System health snapshot"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_threads: int
    open_file_descriptors: int
    network_connections: int
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for production readiness
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Monitoring configuration
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.collection_interval = self.config.get('collection_interval', 60)  # seconds
        self.metrics_retention_hours = self.config.get('metrics_retention_hours', 24)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'response_time_ms': 5000.0,
            'error_rate_percent': 5.0
        })
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.system_health_history = deque(maxlen=1000)
        self.component_metrics = defaultdict(lambda: defaultdict(list))
        self.performance_timers = {}
        self.error_tracking = defaultdict(list)
        
        # Real-time monitoring
        self.monitoring_thread = None
        self.monitoring_lock = threading.Lock()
        self.is_monitoring = False
        
        # Performance analysis
        self.baseline_metrics = {}
        self.anomaly_detection_window = 100
        self.performance_alerts = []
        
        # Component tracking
        self.component_registry = {}
        self.api_endpoint_metrics = defaultdict(lambda: {
            'calls': 0, 'total_time': 0, 'errors': 0, 'last_call': None
        })
        
        # Resource usage tracking
        self.memory_profiling = self.config.get('memory_profiling', True)
        self.detailed_profiling = self.config.get('detailed_profiling', False)
        
        # Statistics
        self.monitoring_stats = {
            'total_metrics_collected': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0,
            'monitoring_uptime_seconds': 0,
            'last_health_check': None
        }
        
        logging.info("Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitoring_enabled and not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logging.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background"""
        start_time = time.time()
        
        while self.is_monitoring:
            try:
                # Collect system health metrics
                health_snapshot = self._collect_system_health()
                self.system_health_history.append(health_snapshot)
                
                # Check for alerts
                self._check_alert_conditions(health_snapshot)
                
                # Detect anomalies
                self._detect_performance_anomalies()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Update monitoring statistics
                self.monitoring_stats['monitoring_uptime_seconds'] = time.time() - start_time
                self.monitoring_stats['last_health_check'] = datetime.now().isoformat()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_health(self) -> SystemHealthSnapshot:
        """Collect current system health metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process information
            current_process = psutil.Process()
            active_threads = current_process.num_threads()
            
            try:
                open_fds = current_process.num_fds()
            except (AttributeError, psutil.AccessDenied):
                open_fds = 0
            
            try:
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                network_connections = 0
            
            # GPU metrics (if available)
            gpu_utilization = None
            gpu_memory_percent = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilization = gpus[0].load * 100
                    gpu_memory_percent = gpus[0].memoryUtil * 100
            except ImportError:
                pass
            
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=(disk.used / disk.total) * 100,
                active_threads=active_threads,
                open_file_descriptors=open_fds,
                network_connections=network_connections,
                gpu_utilization=gpu_utilization,
                gpu_memory_percent=gpu_memory_percent
            )
            
            return snapshot
            
        except Exception as e:
            logging.error(f"Error collecting system health: {e}")
            return SystemHealthSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0, memory_percent=0, disk_usage_percent=0,
                active_threads=0, open_file_descriptors=0, network_connections=0
            )
    
    def record_metric(self, name: str, value: float, unit: str = 'count', 
                     component: str = 'system', category: str = 'general',
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            component: Component that generated the metric
            category: Metric category
            metadata: Additional metadata
        """
        if not self.monitoring_enabled:
            return
        
        with self.monitoring_lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now().isoformat(),
                component=component,
                category=category,
                metadata=metadata or {}
            )
            
            # Store in history
            self.metrics_history[name].append(metric)
            self.component_metrics[component][name].append(metric)
            
            # Update statistics
            self.monitoring_stats['total_metrics_collected'] += 1
    
    def performance_timer(self, component: str = 'unknown', category: str = 'timing'):
        """
        Decorator for timing function execution
        
        Args:
            component: Component name
            category: Metric category
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.monitoring_enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = None
                
                if self.memory_profiling:
                    try:
                        import tracemalloc
                        if not tracemalloc.is_tracing():
                            tracemalloc.start()
                        start_memory = tracemalloc.get_traced_memory()[0]
                    except ImportError:
                        pass
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    self.record_error(func.__name__, e, component)
                    raise
                finally:
                    # Record timing
                    execution_time = (time.time() - start_time) * 1000  # milliseconds
                    self.record_metric(
                        name=f"{func.__name__}_execution_time",
                        value=execution_time,
                        unit='ms',
                        component=component,
                        category=category,
                        metadata={
                            'function_name': func.__name__,
                            'success': success,
                            'error': error
                        }
                    )
                    
                    # Record memory usage if available
                    if self.memory_profiling and start_memory is not None:
                        try:
                            import tracemalloc
                            current_memory = tracemalloc.get_traced_memory()[0]
                            memory_delta = current_memory - start_memory
                            self.record_metric(
                                name=f"{func.__name__}_memory_usage",
                                value=memory_delta / 1024 / 1024,  # MB
                                unit='MB',
                                component=component,
                                category='memory',
                                metadata={'function_name': func.__name__}
                            )
                        except ImportError:
                            pass
                
                return result
            return wrapper
        return decorator
    
    def record_api_call(self, endpoint: str, method: str, status_code: int, 
                       response_time_ms: float, error: Optional[str] = None):
        """
        Record API endpoint performance metrics
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: Response status code
            response_time_ms: Response time in milliseconds
            error: Error message if any
        """
        endpoint_key = f"{method}:{endpoint}"
        
        with self.monitoring_lock:
            metrics = self.api_endpoint_metrics[endpoint_key]
            metrics['calls'] += 1
            metrics['total_time'] += response_time_ms
            metrics['last_call'] = datetime.now().isoformat()
            
            if status_code >= 400 or error:
                metrics['errors'] += 1
            
            # Record detailed metrics
            self.record_metric(
                name=f"api_response_time",
                value=response_time_ms,
                unit='ms',
                component='api',
                category='endpoint',
                metadata={
                    'endpoint': endpoint,
                    'method': method,
                    'status_code': status_code,
                    'error': error
                }
            )
            
            # Record error rate
            error_rate = (metrics['errors'] / metrics['calls']) * 100
            self.record_metric(
                name=f"api_error_rate",
                value=error_rate,
                unit='percent',
                component='api',
                category='reliability',
                metadata={'endpoint': endpoint_key}
            )
    
    def record_error(self, operation: str, error: Exception, component: str = 'unknown'):
        """
        Record error occurrence
        
        Args:
            operation: Operation that failed
            error: Exception that occurred
            component: Component where error occurred
        """
        with self.monitoring_lock:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'component': component,
                'traceback': traceback.format_exc()
            }
            
            self.error_tracking[component].append(error_info)
            
            # Record error metric
            self.record_metric(
                name='error_count',
                value=1,
                unit='count',
                component=component,
                category='errors',
                metadata=error_info
            )
    
    def _check_alert_conditions(self, health_snapshot: SystemHealthSnapshot):
        """Check if any alert conditions are met"""
        alerts = []
        
        # Check CPU usage
        if health_snapshot.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"High CPU usage: {health_snapshot.cpu_percent:.1f}%",
                'threshold': self.alert_thresholds['cpu_percent'],
                'current_value': health_snapshot.cpu_percent
            })
        
        # Check memory usage
        if health_snapshot.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"High memory usage: {health_snapshot.memory_percent:.1f}%",
                'threshold': self.alert_thresholds['memory_percent'],
                'current_value': health_snapshot.memory_percent
            })
        
        # Check disk usage
        if health_snapshot.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append({
                'type': 'disk_high',
                'severity': 'critical',
                'message': f"High disk usage: {health_snapshot.disk_usage_percent:.1f}%",
                'threshold': self.alert_thresholds['disk_usage_percent'],
                'current_value': health_snapshot.disk_usage_percent
            })
        
        # Add alerts to history
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.performance_alerts.append(alert)
            self.monitoring_stats['alerts_generated'] += 1
            logging.warning(f"Performance Alert: {alert['message']}")
    
    def _detect_performance_anomalies(self):
        """Detect performance anomalies using statistical analysis"""
        if len(self.system_health_history) < self.anomaly_detection_window:
            return
        
        recent_snapshots = list(self.system_health_history)[-self.anomaly_detection_window:]
        
        # Analyze CPU usage anomalies
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        cpu_mean = np.mean(cpu_values)
        cpu_std = np.std(cpu_values)
        
        current_cpu = recent_snapshots[-1].cpu_percent
        if abs(current_cpu - cpu_mean) > 2 * cpu_std and cpu_std > 5:
            self.performance_alerts.append({
                'type': 'cpu_anomaly',
                'severity': 'info',
                'message': f"CPU usage anomaly detected: {current_cpu:.1f}% (mean: {cpu_mean:.1f}%, std: {cpu_std:.1f}%)",
                'timestamp': datetime.now().isoformat(),
                'current_value': current_cpu,
                'baseline_mean': cpu_mean,
                'baseline_std': cpu_std
            })
            self.monitoring_stats['anomalies_detected'] += 1
        
        # Analyze memory usage anomalies
        memory_values = [s.memory_percent for s in recent_snapshots]
        memory_mean = np.mean(memory_values)
        memory_std = np.std(memory_values)
        
        current_memory = recent_snapshots[-1].memory_percent
        if abs(current_memory - memory_mean) > 2 * memory_std and memory_std > 5:
            self.performance_alerts.append({
                'type': 'memory_anomaly',
                'severity': 'info',
                'message': f"Memory usage anomaly detected: {current_memory:.1f}% (mean: {memory_mean:.1f}%, std: {memory_std:.1f}%)",
                'timestamp': datetime.now().isoformat(),
                'current_value': current_memory,
                'baseline_mean': memory_mean,
                'baseline_std': memory_std
            })
            self.monitoring_stats['anomalies_detected'] += 1
    
    def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        cutoff_str = cutoff_time.isoformat()
        
        # Clean up metric history
        for metric_name in self.metrics_history:
            while (self.metrics_history[metric_name] and 
                   self.metrics_history[metric_name][0].timestamp < cutoff_str):
                self.metrics_history[metric_name].popleft()
        
        # Clean up system health history
        while (self.system_health_history and 
               self.system_health_history[0].timestamp < cutoff_str):
            self.system_health_history.popleft()
        
        # Clean up alerts
        self.performance_alerts = [
            alert for alert in self.performance_alerts 
            if alert['timestamp'] >= cutoff_str
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.system_health_history:
            return {'error': 'No performance data available'}
        
        recent_snapshots = list(self.system_health_history)[-10:]  # Last 10 snapshots
        
        # Calculate averages
        avg_cpu = np.mean([s.cpu_percent for s in recent_snapshots])
        avg_memory = np.mean([s.memory_percent for s in recent_snapshots])
        avg_disk = np.mean([s.disk_usage_percent for s in recent_snapshots])
        
        # API performance summary
        api_summary = {}
        for endpoint, metrics in self.api_endpoint_metrics.items():
            if metrics['calls'] > 0:
                api_summary[endpoint] = {
                    'total_calls': metrics['calls'],
                    'avg_response_time': metrics['total_time'] / metrics['calls'],
                    'error_rate': (metrics['errors'] / metrics['calls']) * 100,
                    'last_call': metrics['last_call']
                }
        
        # Component performance
        component_summary = {}
        for component, metrics in self.component_metrics.items():
            component_summary[component] = {
                'total_metrics': sum(len(metric_list) for metric_list in metrics.values()),
                'metric_types': list(metrics.keys())
            }
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.performance_alerts[-10:]
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'disk_usage_percent': avg_disk,
                'status': 'healthy' if all([
                    avg_cpu < self.alert_thresholds['cpu_percent'],
                    avg_memory < self.alert_thresholds['memory_percent'],
                    avg_disk < self.alert_thresholds['disk_usage_percent']
                ]) else 'warning'
            },
            'api_performance': api_summary,
            'component_performance': component_summary,
            'recent_alerts': recent_alerts,
            'monitoring_stats': self.monitoring_stats,
            'total_metrics_collected': self.monitoring_stats['total_metrics_collected'],
            'monitoring_uptime': f"{self.monitoring_stats['monitoring_uptime_seconds']:.1f} seconds"
        }
    
    def get_detailed_metrics(self, component: Optional[str] = None, 
                           metric_name: Optional[str] = None,
                           hours_back: int = 1) -> Dict[str, Any]:
        """
        Get detailed metrics for analysis
        
        Args:
            component: Filter by component
            metric_name: Filter by metric name  
            hours_back: Hours of history to return
            
        Returns:
            Detailed metrics data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        cutoff_str = cutoff_time.isoformat()
        
        filtered_metrics = {}
        
        for name, metric_deque in self.metrics_history.items():
            # Filter by metric name if specified
            if metric_name and metric_name != name:
                continue
            
            # Filter by component and time
            relevant_metrics = [
                metric for metric in metric_deque
                if (metric.timestamp >= cutoff_str and
                    (not component or metric.component == component))
            ]
            
            if relevant_metrics:
                filtered_metrics[name] = [asdict(metric) for metric in relevant_metrics]
        
        return {
            'metrics': filtered_metrics,
            'filter': {
                'component': component,
                'metric_name': metric_name,
                'hours_back': hours_back
            },
            'total_metrics': sum(len(metrics) for metrics in filtered_metrics.values())
        }
    
    def get_system_health_trend(self, hours_back: int = 2) -> Dict[str, Any]:
        """Get system health trend over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        cutoff_str = cutoff_time.isoformat()
        
        relevant_snapshots = [
            s for s in self.system_health_history
            if s.timestamp >= cutoff_str
        ]
        
        if not relevant_snapshots:
            return {'error': 'No health data available for specified time range'}
        
        return {
            'time_range': f"Last {hours_back} hours",
            'snapshots': [asdict(snapshot) for snapshot in relevant_snapshots],
            'summary': {
                'avg_cpu': np.mean([s.cpu_percent for s in relevant_snapshots]),
                'max_cpu': max([s.cpu_percent for s in relevant_snapshots]),
                'avg_memory': np.mean([s.memory_percent for s in relevant_snapshots]),
                'max_memory': max([s.memory_percent for s in relevant_snapshots]),
                'avg_disk': np.mean([s.disk_usage_percent for s in relevant_snapshots]),
                'total_snapshots': len(relevant_snapshots)
            }
        }
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'detailed_metrics': self.get_detailed_metrics(hours_back=24),
            'system_health_trend': self.get_system_health_trend(hours_back=24),
            'monitoring_configuration': self.config
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logging.info(f"Performance metrics exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self.monitoring_lock:
            self.metrics_history.clear()
            self.system_health_history.clear()
            self.component_metrics.clear()
            self.api_endpoint_metrics.clear()
            self.error_tracking.clear()
            self.performance_alerts.clear()
            
            self.monitoring_stats = {
                'total_metrics_collected': 0,
                'alerts_generated': 0,
                'anomalies_detected': 0,
                'monitoring_uptime_seconds': 0,
                'last_health_check': None
            }
        
        logging.info("Performance metrics reset")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()

# Global performance monitor instance
_global_monitor = None

def get_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(config)
    return _global_monitor

def monitor_performance(component: str = 'unknown', category: str = 'timing'):
    """Convenience decorator for performance monitoring"""
    monitor = get_performance_monitor()
    return monitor.performance_timer(component, category)