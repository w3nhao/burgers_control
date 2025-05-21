import argparse
import sys
import os
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def list_examples():
    """List all available examples in the directory"""
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    example_files = [
        f for f in os.listdir(examples_dir) 
        if f.endswith('.py') and f != '__main__.py' and not f.startswith('_')
    ]
    
    example_names = [os.path.splitext(f)[0] for f in example_files]
    return example_names, example_files

def main():
    """Main function to run examples"""
    example_names, example_files = list_examples()
    
    # Create command line parser
    parser = argparse.ArgumentParser(description='Run Burgers equation examples')
    parser.add_argument('example', choices=example_names + ['all'], 
                        help='Which example to run (or "all" to run all examples)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.example == 'all':
        print("Running all examples...")
        for name, file in zip(example_names, example_files):
            print(f"\n{'='*50}")
            print(f"RUNNING EXAMPLE: {name}")
            print(f"{'='*50}")
            
            # Import and run the example
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            example_module = import_module_from_path(name, module_path)
            
            print(f"\n{'='*50}")
            print(f"COMPLETED EXAMPLE: {name}")
            print(f"{'='*50}")
    else:
        # Run a single example
        idx = example_names.index(args.example)
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), example_files[idx])
        example_module = import_module_from_path(args.example, module_path)

if __name__ == "__main__":
    main() 