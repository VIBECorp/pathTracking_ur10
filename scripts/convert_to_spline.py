#!/usr/bin/env python
"""
Convert trajectory_kuka.json to Spline format for use with evaluate.py
"""

import json
import numpy as np
import sys
import os

# Add parent directory to path to import tracking modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.utils.spline_utils import Spline


def convert_trajectory_to_spline(input_file, output_dir, output_filename='spline_kuka.json',
                                 length_correction_step_size=2e-4,
                                 use_normalized_length_correction_step_size=False):
    """
    Convert trajectory JSON to Spline format.
    
    Args:
        input_file: Path to input trajectory JSON file (with 'joint' key)
        output_dir: Directory to save spline file
        output_filename: Name of output spline file
        length_correction_step_size: Spline length correction step size
        use_normalized_length_correction_step_size: Whether to use normalized step size
    """
    
    # Load trajectory data
    print(f"Loading trajectory from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract joint positions
    joint_data = np.array(data['joint'])  # Shape: (num_waypoints, num_joints)
    num_waypoints, num_joints = joint_data.shape
    
    print(f"Trajectory: {num_waypoints} waypoints, {num_joints} joints")
    
    # Convert to spline format (transpose: joints x waypoints)
    curve_data = joint_data.T  # Shape: (num_joints, num_waypoints)
    
    # Create spline from trajectory
    print("Creating spline from trajectory...")
    spline = Spline(
        curve_data=curve_data,
        curve_data_slicing_step=1,
        length_correction_step_size=length_correction_step_size,
        use_normalized_length_correction_step_size=use_normalized_length_correction_step_size,
        method="auto",
        curvature_at_ends=0
    )
    
    print(f"Spline length: {spline.get_length():.4f}")
    print(f"Max distance between knots: {spline.max_dis_between_knots:.4f}")

    # Compute execution time (seconds) from trajectory time if available
    execution_time = None
    if 'time' in data and len(data['time']) > 0:
        time_array = np.array(data['time'], dtype=float)
        execution_time = (time_array[-1] - time_array[0]) * 1e-9  # ns -> s
        print(f"Execution time from input trajectory: {execution_time:.6f} s")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save spline
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving spline to {output_path}...")
    
    # Use save_to_json which matches the format of sample files
    spline.save_to_json(output_path, make_dir=True)

    # Append execution_time to saved spline (keep JSON format compatible with sample)
    if execution_time is not None:
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        saved_data['execution_time'] = execution_time
        with open(output_path, 'w') as f:
            f.write(json.dumps(saved_data, sort_keys=True))
            f.flush()
        print("Added execution_time to saved spline.")
    
    # Verify the saved file format matches sample
    with open(output_path, 'r') as f:
        saved_data = json.load(f)
    
    # Check that all required keys are present (matching sample format)
    required_keys = ['_length_correction_step_size', '_use_normalized_length_correction_step_size',
                     '_curve_data_slicing_step', '_u', '_method', '_curvature_at_ends',
                     '_uncorrected_curve_length', '_b_spline', '_tck']
    
    missing_keys = [key for key in required_keys if key not in saved_data]
    if missing_keys:
        print(f"Warning: Missing keys in saved file: {missing_keys}")
    else:
        print("Saved spline format matches sample format ✓")
    
    print("Conversion complete!")
    return spline, output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert trajectory to Spline format')
    parser.add_argument('--input', type=str, default='temp/trajectory_kuka.json',
                        help='Input trajectory JSON file')
    parser.add_argument('--output_dir', type=str, default='temp/splines',
                        help='Output directory for spline file')
    parser.add_argument('--output_filename', type=str, default='spline_kuka.json',
                        help='Output spline filename')
    parser.add_argument('--length_correction_step_size', type=float, default=2e-4,
                        help='Spline length correction step size')
    parser.add_argument('--use_normalized_length_correction_step_size', action='store_true',
                        help='Use normalized length correction step size')
    
    args = parser.parse_args()
    
    convert_trajectory_to_spline(
        input_file=args.input,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        length_correction_step_size=args.length_correction_step_size,
        use_normalized_length_correction_step_size=args.use_normalized_length_correction_step_size
    )
