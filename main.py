import gc
import asyncio  
from src.initialization_LLMs import load_all_models
from src.processes import process_model
from src.dataset_func.load_dataset import load_dataset
from src.dataset_func.process_dataset import process_dataset
from src.dataset_func.batches import prepare_batches
from src.analysis import perform_result_analysis
from src.resources.model_names import model_names

# PARAMETERS
BATCH_SIZE = 8
SEED = 42
RANKING_WINDOW = 2
MODEL_BATCH = 1  # Number of models to process in parallel (batch mode)

BATCH_PREPARATION = True
MODEL_LOADING = True
ANALYSIS = True
RUN_MODE = 'single'  # 'single' for single model mode, 'batch' for batch mode

#######################################################

if __name__ == "__main__":

    if BATCH_PREPARATION:
        # Load the dataset directory
        dataset_dir = load_dataset()

        # Process the dataset and get the DataFrame
        try:
            processed_dataset, genre_occurrences = process_dataset(dataset_dir, seed=SEED)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing dataset: {e}")
            exit(1)

        # Prepare batches for model processing
        batches, batches_info, batches_paragon = prepare_batches(
            batch_size=BATCH_SIZE, seed=SEED, df=processed_dataset
        )
        print(f" number of batches = {len(batches)}")
        exit(1)

    if MODEL_LOADING:
        if RUN_MODE == 'single':
            # Single model mode: Load and process one model at a time without asyncio
            for model_name in model_names:
                print(f"Loading and processing model: {model_name}")

                # Load the model and tokenizer for the current model
                tokenizers, models = load_all_models([model_name])
                tokenizer = tokenizers[model_name]
                model = models[model_name]

                try:
                    # Use asyncio.run() to run the async process_model function in a synchronous context
                    asyncio.run(process_model(
                        model_name=model_name,
                        model=model,
                        tokenizer=tokenizer,
                        batches=batches,
                        batches_info=batches_info,
                        batches_paragon=batches_paragon,
                        ranking_window=RANKING_WINDOW
                    ))
                except Exception as e:
                    print(f"Error processing model {model_name}: {e}")

                # Free memory after processing the model
                del tokenizer, model, tokenizers, models
                gc.collect()  # Trigger garbage collection

        elif RUN_MODE == 'batch':
            # Batch mode: Process models in batches using asyncio

            async def batch_process():
                tasks = []
                semaphore = asyncio.Semaphore(MODEL_BATCH)  # Controls concurrent execution

                async def limited_process_model(model_name, model, tokenizer):
                    async with semaphore:
                        try:
                            # Schedule the processing task for each model
                            return await process_model(
                                model_name=model_name,
                                model=model,
                                tokenizer=tokenizer,
                                batches=batches,
                                batches_info=batches_info,
                                batches_paragon=batches_paragon,
                                ranking_window=RANKING_WINDOW
                            )
                        except Exception as e:
                            print(f"Error processing model {model_name}: {e}")
                            return None

                # Split model_names into batches
                model_batches = [model_names[i:i + MODEL_BATCH] for i in range(0, len(model_names), MODEL_BATCH)]

                # Process each batch of models
                for batch in model_batches:
                    # Load models for the current batch
                    tokenizers, models = load_all_models(batch)

                    # Process each model in the current batch
                    for model_name in batch:
                        tokenizer = tokenizers[model_name]
                        model = models[model_name]

                        # Schedule the processing task using the semaphore
                        task = limited_process_model(model_name=model_name, model=model, tokenizer=tokenizer)
                        tasks.append(task)

                    # Run tasks for the current batch
                    await asyncio.gather(*tasks)

                    # Free memory after processing the batch
                    del tokenizers, models
                    gc.collect()  # Trigger garbage collection

            # Run batch processing asynchronously
            asyncio.run(batch_process())

    if ANALYSIS:
        # Perform result analysis if enabled
        perform_result_analysis(genre_occurrences=genre_occurrences)

    print("All results have been saved.")
