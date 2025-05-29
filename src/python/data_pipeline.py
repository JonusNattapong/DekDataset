import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    input_sources: List[str]
    output_format: str
    batch_size: int = 10
    max_workers: int = 4
    quality_threshold: float = 0.8
    retry_attempts: int = 3
    cache_enabled: bool = True
    
class DataPipeline:
    """Advanced data processing pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processors = []
        self.cache = {}
        self.metrics = {
            "processed": 0,
            "failed": 0,
            "cached_hits": 0,
            "start_time": None
        }
    
    def add_processor(self, processor: Callable[[Dict], Dict]):
        """Add a data processor to the pipeline"""
        self.processors.append(processor)
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of data items"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit batch items for processing
            future_to_item = {
                executor.submit(self._process_item, item): item 
                for item in batch
            }
            
            # Collect results
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.metrics["processed"] += 1
                    else:
                        self.metrics["failed"] += 1
                except Exception as e:
                    print(f"[ERROR] Processing failed: {e}")
                    self.metrics["failed"] += 1
        
        return results
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item through all processors"""
        try:
            # Check cache first
            if self.config.cache_enabled:
                item_key = self._get_cache_key(item)
                if item_key in self.cache:
                    self.metrics["cached_hits"] += 1
                    return self.cache[item_key]
            
            # Apply all processors in sequence
            result = item.copy()
            for processor in self.processors:
                result = processor(result)
                if not result:  # Processor rejected the item
                    return None
            
            # Cache result
            if self.config.cache_enabled:
                self.cache[item_key] = result
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Item processing failed: {e}")
            return None
    
    def _get_cache_key(self, item: Dict) -> str:
        """Generate cache key for item"""
        return str(hash(str(sorted(item.items()))))
    
    async def run(self, data: List[Dict]) -> List[Dict]:
        """Run the complete pipeline"""
        self.metrics["start_time"] = time.time()
        
        print(f"[INFO] Starting pipeline with {len(data)} items")
        print(f"[INFO] Batch size: {self.config.batch_size}, Workers: {self.config.max_workers}")
        
        all_results = []
        
        # Process data in batches
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            
            print(f"[INFO] Processing batch {i//self.config.batch_size + 1}/{(len(data)-1)//self.config.batch_size + 1}")
            
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Progress update
            progress = (i + len(batch)) / len(data) * 100
            print(f"[INFO] Progress: {progress:.1f}% ({len(all_results)} processed)")
        
        # Final metrics
        elapsed_time = time.time() - self.metrics["start_time"]
        print(f"\n[INFO] Pipeline completed in {elapsed_time:.2f}s")
        print(f"[INFO] Processed: {self.metrics['processed']}")
        print(f"[INFO] Failed: {self.metrics['failed']}")
        print(f"[INFO] Cache hits: {self.metrics['cached_hits']}")
        print(f"[INFO] Success rate: {self.metrics['processed']/(self.metrics['processed']+self.metrics['failed'])*100:.1f}%")
        
        return all_results

# Standard processors
def text_cleaner(item: Dict) -> Dict:
    """Clean text fields"""
    for key, value in item.items():
        if isinstance(value, str):
            # Remove extra whitespace
            value = ' '.join(value.split())
            # Remove empty strings
            if not value.strip():
                return None
            item[key] = value
    return item

def length_validator(min_length: int = 10, max_length: int = 1000):
    """Create length validator"""
    def validator(item: Dict) -> Dict:
        for key, value in item.items():
            if isinstance(value, str):
                if len(value) < min_length or len(value) > max_length:
                    return None
        return item
    return validator

def duplicate_remover():
    """Remove duplicate items"""
    seen = set()
    
    def remover(item: Dict) -> Dict:
        # Create signature for the item
        signature = str(sorted(item.items()))
        if signature in seen:
            return None
        seen.add(signature)
        return item
    
    return remover

def create_standard_pipeline(config: PipelineConfig) -> DataPipeline:
    """Create a standard data processing pipeline"""
    pipeline = DataPipeline(config)
    
    # Add standard processors
    pipeline.add_processor(text_cleaner)
    pipeline.add_processor(length_validator(min_length=5, max_length=2000))
    pipeline.add_processor(duplicate_remover())
    
    return pipeline
