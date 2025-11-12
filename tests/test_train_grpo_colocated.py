"""
Unit tests for train_grpo_ray_colocated.py

Tests the colocated GRPO training components using adapter pattern
following the style of existing tests.
"""

import torch
import pytest
from typing import List

from .adapters_colocated import (
    run_create_trajectory as create_trajectory,
    run_generator_compute_advantages as compute_advantages,
    run_compute_trajectory_statistics as compute_trajectory_statistics,
    run_mock_training_step as mock_training_step,
    run_validate_colocated_worker_config as validate_config,
    run_test_ray_actor_creation as test_ray_actor_creation,
    run_batch_prompts_processing as batch_prompts_processing,
)


def test_create_trajectory_with_values(
    numpy_snapshot,
):
    """Test Trajectory creation with all components including values"""
    prompts = ["Write a story including apple", "Write a poem including ocean", "Write a dialogue including friendship", "Write an essay including technology"]
    responses = ["A story about red apples", "Ocean waves crash poetically", "Friends talked happily together", "Technology advances rapidly today"]
    rewards = torch.tensor([1.0, 0.8, 0.9, 0.7])
    log_probs = torch.tensor([-0.1, -0.3, -0.2, -0.4])
    values = torch.tensor([0.5, 0.4, 0.45, 0.35])
    
    output = create_trajectory(
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        log_probs=log_probs,
        values=values
    )
    
    numpy_snapshot.assert_match(output)


def test_create_trajectory_without_values(
    numpy_snapshot,
):
    """Test Trajectory creation without optional values"""
    prompts = ["Write a story including mountain", "Write a poem including sunset", "Write a dialogue including adventure", "Write an essay including education"]
    responses = ["Mountains rise tall and proud", "Sunset paints the sky golden", "Adventure awaits brave souls", "Education opens many doors"]
    rewards = torch.tensor([1.0, 0.8, 0.9, 0.7])
    log_probs = torch.tensor([-0.1, -0.3, -0.2, -0.4])
    
    output = create_trajectory(
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        log_probs=log_probs,
        values=None
    )
    
    numpy_snapshot.assert_match(output)


def test_generator_compute_advantages_single_group(
    numpy_snapshot,
):
    """Test advantage computation for a single group of responses"""
    torch.manual_seed(42)
    # Create rewards for one group (G=4 responses)
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.3])
    
    advantages = compute_advantages(rewards, group_size=4)
    
    numpy_snapshot.assert_match(advantages)


def test_generator_compute_advantages_multiple_groups(
    numpy_snapshot,
):
    """Test advantage computation for multiple groups"""
    torch.manual_seed(42)
    # Create rewards for 3 groups (12 total responses)
    rewards = torch.tensor([
        1.0, 0.5, 0.8, 0.3,  # Group 1
        0.9, 0.7, 0.6, 0.4,  # Group 2  
        0.8, 0.9, 0.5, 0.7   # Group 3
    ])
    
    advantages = compute_advantages(rewards, group_size=4)
    
    numpy_snapshot.assert_match(advantages)


def test_compute_trajectory_statistics_multiple_trajectories(
    numpy_snapshot,
):
    """Test computation of statistics across multiple trajectories"""
    torch.manual_seed(42)
    # Create rewards from 3 trajectories
    trajectories_rewards = [
        torch.tensor([1.0, 0.8, 0.9, 0.7]),  # Trajectory 1
        torch.tensor([0.6, 0.5, 0.8, 0.9]),  # Trajectory 2
        torch.tensor([0.7, 0.9, 0.6, 0.8])   # Trajectory 3
    ]
    
    stats = compute_trajectory_statistics(trajectories_rewards)
    
    numpy_snapshot.assert_match(stats)


def test_compute_trajectory_statistics_empty_list(
    numpy_snapshot,
):
    """Test statistics computation with empty trajectory list"""
    trajectories_rewards = []
    
    stats = compute_trajectory_statistics(trajectories_rewards)
    
    numpy_snapshot.assert_match(stats)


def test_mock_training_step_single_prompt(
    numpy_snapshot,
):
    """Test training step with single prompt"""
    torch.manual_seed(42)
    prompts = ["Write a story including dragon"]
    
    result = mock_training_step(prompts, step_count=0)
    
    numpy_snapshot.assert_match(result)


def test_mock_training_step_multiple_prompts(
    numpy_snapshot,
):
    """Test training step with multiple prompts"""
    torch.manual_seed(42)
    prompts = [
        "Write a story including rainbow",
        "Write a poem including forest", 
        "Write a dialogue including mystery",
        "Write an essay including innovation"
    ]
    
    result = mock_training_step(prompts, step_count=5)
    
    numpy_snapshot.assert_match(result)


def test_validate_colocated_worker_config(
    numpy_snapshot,
):
    """Test validation of ColocatedWorker configuration"""
    config = validate_config()
    
    numpy_snapshot.assert_match(config)


def test_ray_actor_creation_single_worker(
    numpy_snapshot,
):
    """Test Ray actor creation with single worker"""
    result = test_ray_actor_creation(num_workers=1)
    
    # Remove variable fields for consistent testing
    stable_result = {
        'workers_created': result['workers_created'],
        'workers_responsive': result['workers_responsive'],
        'all_workers_initialized': result['all_workers_initialized'],
        'actor_creation_success': result['actor_creation_success']
    }
    
    numpy_snapshot.assert_match(stable_result)


def test_ray_actor_creation_multiple_workers(
    numpy_snapshot,
):
    """Test Ray actor creation with multiple workers"""
    result = test_ray_actor_creation(num_workers=2)
    
    # Remove variable fields for consistent testing
    stable_result = {
        'workers_created': result['workers_created'],
        'workers_responsive': result['workers_responsive'],
        'all_workers_initialized': result['all_workers_initialized'],
        'actor_creation_success': result['actor_creation_success']
    }
    
    numpy_snapshot.assert_match(stable_result)


def test_batch_prompts_processing_small_batch(
    numpy_snapshot,
):
    """Test batch processing with small batch size"""
    torch.manual_seed(42)
    prompts = [
        "Write a story including butterfly",
        "Write a poem including winter",
        "Write a dialogue including courage",
        "Write an essay including creativity",
        "Write a letter including gratitude"
    ]
    
    result = batch_prompts_processing(prompts, batch_size=2)
    
    # Extract only numeric data for snapshot comparison
    numeric_result = {
        'total_prompts': result['total_prompts'],
        'num_batches': result['num_batches'],
        'avg_batch_size': result['avg_batch_size'],
        'processing_complete': result['processing_complete']
    }
    
    numpy_snapshot.assert_match(numeric_result)


def test_batch_prompts_processing_large_batch(
    numpy_snapshot,
):
    """Test batch processing with large batch size"""
    torch.manual_seed(42)
    prompts = [
        "Write a story including castle",
        "Write a poem including starlight", 
        "Write a dialogue including wisdom"
    ]
    
    result = batch_prompts_processing(prompts, batch_size=5)
    
    # Extract only numeric data for snapshot comparison
    numeric_result = {
        'total_prompts': result['total_prompts'],
        'num_batches': result['num_batches'],
        'avg_batch_size': result['avg_batch_size'],
        'processing_complete': result['processing_complete']
    }
    
    numpy_snapshot.assert_match(numeric_result)


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_advantages_computation_different_group_sizes(
    numpy_snapshot,
    group_size,
):
    """Test advantage computation with different group sizes"""
    torch.manual_seed(42)
    # Create rewards for 2 groups
    num_groups = 2
    rewards = torch.rand(num_groups * group_size)
    
    advantages = compute_advantages(rewards, group_size=group_size)
    
    output = {
        'advantages': advantages,
        'group_size': group_size,
        'num_groups': num_groups,
        'total_responses': len(rewards)
    }
    
    numpy_snapshot.assert_match(output)


@pytest.mark.parametrize("num_trajectories", [1, 3, 5])
def test_trajectory_statistics_scaling(
    numpy_snapshot,
    num_trajectories,
):
    """Test trajectory statistics with different numbers of trajectories"""
    torch.manual_seed(42)
    trajectories_rewards = []
    
    for i in range(num_trajectories):
        # Each trajectory has G=4 responses
        rewards = torch.rand(4)
        trajectories_rewards.append(rewards)
    
    stats = compute_trajectory_statistics(trajectories_rewards)
    
    numpy_snapshot.assert_match(stats)
