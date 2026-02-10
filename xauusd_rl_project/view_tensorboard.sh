#!/bin/bash
# Start TensorBoard to view training progress
# Usage: ./view_tensorboard.sh

echo "Starting TensorBoard..."
echo "View training logs at: http://localhost:6006"
echo "Press Ctrl+C to stop"
echo ""

tensorboard --logdir logs/tensorboard --port 6006

