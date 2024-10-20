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
MODEL_BATCH = 1  # This controls how many models run concurrently (set to 1 for sequential execution)

BATCH_PREPARATION = True
MODEL_LOADING = True
ANALYSIS = True

#######################################################

if __name__ == "__main__":

    if BATCH_PREPARATION:
        # Load the dataset directory
        dataset_dir = load_dataset()

        # Process the dataset and get the DataFrame
        try:
            processed_dataset, genre_occurrencies = process_dataset(dataset_dir, seed=SEED)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing dataset: {e}")
            exit(1)

        # Prepare batches for model processing
        batches, batches_info, batches_paragon = prepare_batches(
            batch_size=BATCH_SIZE, seed=SEED, df=processed_dataset
        )

    # Define the main asynchronous function
    async def main():

        if MODEL_LOADING:
            tasks = []
            semaphore = asyncio.Semaphore(MODEL_BATCH)  # Controls concurrent execution (1 at a time here)

            async def limited_process_model(model_name, model, tokenizer):
                async with semaphore:  # Ensures limited concurrent execution
                    try:
                        result = await process_model(
                            model_name=model_name,
                            model=model,
                            tokenizer=tokenizer,
                            batches=batches,
                            batches_info=batches_info,
                            batches_paragon=batches_paragon,
                            ranking_window=RANKING_WINDOW
                        )
                        return result
                    except Exception as e:
                        print(f"Error processing model {model_name}: {e}")
                        return None

            # Iterate over each model, loading and processing them one by one
            for model_name in model_names:
                # Load model and tokenizer
                tokenizers, models = load_all_models([model_name])  # Load a single model at a time
                tokenizer = tokenizers[model_name]
                model = models[model_name]

                # Schedule the task using the semaphore
                task = limited_process_model(model_name=model_name, model=model, tokenizer=tokenizer)
                tasks.append(task)

            # Run all tasks sequentially (or limited to MODEL_BATCH at a time)
            all_results = await asyncio.gather(*tasks)

        if ANALYSIS:
            # Perform result analysis if enabled
            perform_result_analysis(genre_occurrences=genre_occurrencies)

        print("All results have been saved.")

    # Run the main function
    asyncio.run(main())
