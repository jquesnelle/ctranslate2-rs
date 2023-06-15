#pragma once

#include <chrono>
#include <future>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "ctranslate2/replica_pool.h"
#include "ctranslate2/generator.h"

#include "rust/cxx.h"

class GeneratorWrapper;

#include "ctranslate2-rs/src/lib.rs.h"

using StringOrMap = std::variant<std::string, std::unordered_map<std::string, std::string>>;
using Tokens = std::vector<std::string>;
using Ids = std::vector<size_t>;
using BatchTokens = std::vector<Tokens>;
using BatchIds = std::vector<Ids>;
using EndToken = std::variant<std::string, std::vector<std::string>, std::vector<size_t>>;

class ComputeTypeResolver
{
private:
    const std::string _device;

public:
    ComputeTypeResolver(std::string device)
        : _device(std::move(device))
    {
    }

    ctranslate2::ComputeType
    operator()(const std::string &compute_type) const
    {
        return ctranslate2::str_to_compute_type(compute_type);
    }

    ctranslate2::ComputeType
    operator()(const std::unordered_map<std::string, std::string> &compute_type) const
    {
        auto it = compute_type.find(_device);
        if (it == compute_type.end())
            return ctranslate2::ComputeType::DEFAULT;
        return operator()(it->second);
    }
};

class DeviceIndexResolver
{
public:
    std::vector<int> operator()(int device_index) const
    {
        return {device_index};
    }

    std::vector<int> operator()(const std::vector<int> &device_index) const
    {
        return device_index;
    }
};

template <typename T>
class AsyncResult
{
public:
    AsyncResult(std::future<T> future)
        : _future(std::move(future))
    {
    }

    const T &result()
    {
        if (!_done)
        {
            {
                py::gil_scoped_release release;
                try
                {
                    _result = _future.get();
                }
                catch (...)
                {
                    _exception = std::current_exception();
                }
            }
            _done = true; // Assign done attribute while the GIL is held.
        }
        if (_exception)
            std::rethrow_exception(_exception);
        return _result;
    }

    bool done()
    {
        constexpr std::chrono::seconds zero_sec(0);
        return _done || _future.wait_for(zero_sec) == std::future_status::ready;
    }

private:
    std::future<T> _future;
    T _result;
    bool _done = false;
    std::exception_ptr _exception;
};

template <typename Result>
std::vector<Result> wait_on_futures(std::vector<std::future<Result>> futures)
{
    std::vector<Result> results;
    results.reserve(futures.size());
    for (auto &future : futures)
        results.emplace_back(future.get());
    return results;
}

template <typename Result>
std::variant<std::vector<Result>, std::vector<AsyncResult<Result>>>
maybe_wait_on_futures(std::vector<std::future<Result>> futures, bool asynchronous)
{
    if (asynchronous)
    {
        std::vector<AsyncResult<Result>> results;
        results.reserve(futures.size());
        for (auto &future : futures)
            results.emplace_back(std::move(future));
        return std::move(results);
    }
    else
    {
        return wait_on_futures(std::move(futures));
    }
}

template <typename T>
class ReplicaPoolHelper
{
public:
    ReplicaPoolHelper(const std::string &model_path,
                      const std::string &device,
                      const std::variant<int, std::vector<int>> &device_index,
                      const StringOrMap &compute_type,
                      size_t inter_threads,
                      size_t intra_threads,
                      long max_queued_batches)
        : _model_loader(std::make_shared<ctranslate2::models::ModelFileReader>(model))
    {
        _model_loader.device = str_to_device(device);
        _model_loader.device_indices = std::visit(DeviceIndexResolver(), device_index);
        _model_loader.compute_type = std::visit(ComputeTypeResolver(device), compute_type);
        _model_loader.num_replicas_per_device = inter_threads;

        _pool_config.num_threads_per_replica = intra_threads;
        _pool_config.max_queued_batches = max_queued_batches;

        _pool = std::make_unique<T>(_model_loader, _pool_config);
    }

    ~ReplicaPoolHelper()
    {
        pybind11::gil_scoped_release nogil;
        _pool.reset();
    }

    std::string device() const
    {
        return device_to_str(_model_loader.device);
    }

    const std::vector<int> &device_index() const
    {
        return _model_loader.device_indices;
    }

    size_t num_replicas() const
    {
        return _pool->num_replicas();
    }

    size_t num_queued_batches() const
    {
        return _pool->num_queued_batches();
    }

    size_t num_active_batches() const
    {
        return _pool->num_active_batches();
    }

protected:
    std::unique_ptr<T> _pool;
    ctranslate2::models::ModelLoader _model_loader;
    ctranslate2::ReplicaPoolConfig _pool_config;
};

class GeneratorWrapper : public ReplicaPoolHelper<ctranslate2::Generator>
{
public:
    using ReplicaPoolHelper::ReplicaPoolHelper;

    std::variant<std::vector<ctranslate2::GenerationResult>,
                 std::vector<AsyncResult<ctranslate2::GenerationResult>>>
    generate_batch(const BatchTokens &tokens,
                   size_t max_batch_size,
                   const std::string &batch_type_str,
                   bool asynchronous,
                   size_t beam_size,
                   float patience,
                   size_t num_hypotheses,
                   float length_penalty,
                   float repetition_penalty,
                   size_t no_repeat_ngram_size,
                   bool disable_unk,
                   const std::optional<std::vector<std::vector<std::string>>> &suppress_sequences,
                   const std::optional<EndToken> &end_token,
                   bool return_end_token,
                   size_t max_length,
                   size_t min_length,
                   const std::optional<std::vector<std::string>> &static_prompt,
                   bool cache_static_prompt,
                   bool include_prompt_in_result,
                   bool return_scores,
                   bool return_alternatives,
                   float min_alternative_expansion_prob,
                   size_t sampling_topk,
                   float sampling_topp,
                   float sampling_temperature,
                   std::function<bool(ctranslate2::GenerationStepResult)> callback)
    {
        if (tokens.empty())
            return {};

        ctranslate2::BatchType batch_type = ctranslate2::str_to_batch_type(batch_type_str);
        ctranslate2::GenerationOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.disable_unk = disable_unk;
        options.sampling_topk = sampling_topk;
        options.sampling_topp = sampling_topp;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.min_length = min_length;
        options.num_hypotheses = num_hypotheses;
        options.return_end_token = return_end_token;
        options.return_scores = return_scores;
        options.return_alternatives = return_alternatives;
        options.cache_static_prompt = cache_static_prompt;
        options.include_prompt_in_result = include_prompt_in_result;
        options.min_alternative_expansion_prob = min_alternative_expansion_prob;
        options.callback = std::move(callback);
        if (suppress_sequences)
            options.suppress_sequences = suppress_sequences.value();
        if (end_token)
            options.end_token = end_token.value();
        if (static_prompt)
            options.static_prompt = static_prompt.value();

        auto futures = _pool->generate_batch_async(tokens, options, max_batch_size, batch_type);
        return maybe_wait_on_futures(std::move(futures), asynchronous);
    }
};