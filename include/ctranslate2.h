#pragma once

#include <chrono>
#include <future>
#include <variant>
#include <string>
#include <vector>

#include "rust/cxx.h"

#include "ctranslate2/replica_pool.h"
#include "ctranslate2/generator.h"

class GeneratorWrapper;
template <class CPPType, class RustType>
class VecVec;
typedef VecVec<std::string, rust::String> VecVecString;
typedef VecVec<size_t, size_t> VecVecUsize;
#include "ctranslate2-rs/src/lib.rs.h"

template <class FromVectorType, class ToVectorType>
static ToVectorType ConvertVector(const FromVectorType &vec)
{
    ToVectorType ret;
    ret.reserve(vec.size());
    for (const auto &item : vec)
        ret.emplace_back(item);
    return ret;
}

template <class FromVectorType, class ToVectorType>
static ToVectorType ConvertVector(FromVectorType &&vec)
{
    ToVectorType ret;
    ret.reserve(vec.size());
    for (auto &item : vec)
        ret.emplace_back(std::move(item));
    return ret;
}

template <class CPPType, class RustType>
class VecVec
{
public:
    using DataType = std::vector<std::vector<CPPType>>;

    VecVec() = default;
    VecVec(const VecVec &) = default;
    VecVec(VecVec &&) = default;
    ~VecVec() = default;
    VecVec &operator=(const VecVec &) = default;
    VecVec &operator=(VecVec &&) = default;

    VecVec(const rust::Vec<RustType> &rhs) : mData(ConvertVector(rhs)) {}
    VecVec(rust::Vec<RustType> &&rhs) : mData(ConvertVector(std::move(rhs))) {}
    VecVec(const DataType &rhs) : mData(rhs) {}
    VecVec(DataType &&rhs) : mData(std::move(rhs)) {}

    rust::Vec<RustType> at(size_t index) const
    {
        rust::Vec<RustType> ret;
        for (const auto &item : mData.at(index))
            ret.push_back(item);
        return ret;
    }

    void push_back(rust::Vec<RustType> data)
    {
        mData.emplace_back(ConvertVector<rust::Vec<RustType>, std::vector<CPPType>>(data));
    }

    void clear() { mData.clear(); }
    void reserve(size_t sz) { mData.reserve(sz); }

    size_t len() const { return mData.size(); }
    bool empty() const { return mData.empty(); }
    const DataType &data() const { return mData; }

private:
    DataType mData;
};

std::unique_ptr<VecVecString> new_vec_vec_string()
{
    return std::make_unique<VecVecString>(VecVecString());
}

std::unique_ptr<VecVecUsize> new_vec_vec_usize()
{
    return std::make_unique<VecVecUsize>(VecVecUsize());
}

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
                      const std::vector<int> &device_indices,
                      const std::string &compute_type,
                      size_t inter_threads,
                      size_t intra_threads,
                      int max_queued_batches)
        : _model_loader(std::make_shared<ctranslate2::models::ModelFileReader>(model_path))
    {
        _model_loader.device = ctranslate2::str_to_device(device);
        _model_loader.device_indices = device_indices;
        _model_loader.compute_type = ctranslate2::str_to_compute_type(compute_type);
        _model_loader.num_replicas_per_device = inter_threads;

        _pool_config.num_threads_per_replica = intra_threads;
        _pool_config.max_queued_batches = (long)max_queued_batches;

        _pool = std::make_unique<T>(_model_loader, _pool_config);
    }

    ~ReplicaPoolHelper()
    {
        _pool.reset();
    }

    rust::String device() const
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
    using CallbackFunction = rust::Fn<bool(GenerationStepResult, GenerateCallbackContext const &)>;
    using Callback = std::pair<CallbackFunction, rust::Box<GenerateCallbackContext>>;

    rust::Vec<GenerationResult> generate_batch(std::unique_ptr<VecVecString> tokens,
                                               size_t max_batch_size,
                                               rust::Str batch_type_str,
                                               rust::Box<GenerationOptions> options) const
    {
        auto futures = _generate_batch_async(std::move(tokens), max_batch_size, std::move(batch_type_str), ConvertGenerationOptions(std::move(options)));
        auto results = wait_on_futures(std::move(futures));
        return ConvertGenerationResults(std::move(results));
    }

    rust::Vec<GenerationResult> generate_batch_with_callback(std::unique_ptr<VecVecString> tokens,
                                                             size_t max_batch_size,
                                                             rust::Str batch_type_str,
                                                             rust::Box<GenerationOptions> options,
                                                             CallbackFunction callback,
                                                             rust::Box<GenerateCallbackContext> context) const
    {
        auto converted = ConvertGenerationOptions(std::move(options));
        converted.callback = [callback, rawContext = context.into_raw()](ctranslate2::GenerationStepResult result) -> bool
        {
            GenerationStepResult converted;
            converted.batch_id = result.batch_id;
            converted.is_last = result.is_last;
            converted.log_prob = result.log_prob ? *result.log_prob : 0;
            converted.log_prob_valid = result.log_prob.has_value();
            converted.step = result.step;
            converted.token_id = result.token_id;

            return callback(std::move(converted), *rawContext);
        };

        auto futures = _generate_batch_async(std::move(tokens), max_batch_size, std::move(batch_type_str), std::move(converted));
        auto results = wait_on_futures(std::move(futures));
        return ConvertGenerationResults(std::move(results));
    }

private:
    std::vector<std::future<ctranslate2::GenerationResult>> _generate_batch_async(std::unique_ptr<VecVecString> tokens,
                                                                                  size_t max_batch_size,
                                                                                  rust::Str batch_type_str,
                                                                                  ctranslate2::GenerationOptions &&options) const
    {
        if (!tokens || tokens->empty())
            return std::vector<std::future<ctranslate2::GenerationResult>>{};

        ctranslate2::BatchType batch_type =
            ctranslate2::str_to_batch_type((std::string)batch_type_str);
        return _pool->generate_batch_async(
            tokens->data(), std::move(options), max_batch_size, batch_type);
    }

    static ctranslate2::GenerationOptions ConvertGenerationOptions(rust::Box<GenerationOptions> options)
    {
        ctranslate2::GenerationOptions ret;
        ret.beam_size = options->beam_size;
        ret.cache_static_prompt = options->cache_static_prompt;
        ret.disable_unk = options->disable_unk;
        ret.end_token = ConvertVector<rust::Vec<rust::String>, std::vector<std::string>>(options->end_token);
        ret.include_prompt_in_result = options->include_prompt_in_result;
        ret.length_penalty = options->length_penalty;
        ret.max_length = options->max_length;
        ret.min_alternative_expansion_prob = options->min_alternative_expansion_prob;
        ret.no_repeat_ngram_size = options->no_repeat_ngram_size;
        ret.num_hypotheses = options->num_hypotheses;
        ret.patience = options->patience;
        ret.repetition_penalty = options->repetition_penalty;
        ret.return_alternatives = options->return_alternatives;
        ret.return_end_token = options->return_end_token;
        ret.return_scores = options->return_scores;
        ret.sampling_temperature = options->sampling_temperature;
        ret.sampling_topk = options->sampling_topk;
        ret.sampling_topp = options->sampling_topp;
        ret.static_prompt = ConvertVector<rust::Vec<rust::String>, std::vector<std::string>>(options->static_prompt);
        ret.suppress_sequences = options->suppress_sequences
                                     ? options->suppress_sequences->data()
                                     : std::vector<std::vector<std::string>>();
        return ret;
    }

    static rust::Vec<GenerationResult> ConvertGenerationResults(std::vector<ctranslate2::GenerationResult> &&results)
    {
        rust::Vec<GenerationResult> ret;
        for (auto &result : results)
            ret.emplace_back(GenerationResult{
                std::make_unique<VecVecString>(VecVecString(std::move(result.sequences))),
                std::make_unique<VecVecUsize>(VecVecUsize(std::move(result.sequences_ids))),
                ConvertVector<std::vector<float>, rust::Vec<float>>(std::move(result.scores))});
        return ret;
    }
};

std::unique_ptr<GeneratorWrapper> new_generator_wrapper(
    rust::Str model_path,
    rust::Str device,
    rust::Vec<int> device_indicies,
    rust::Str compute_type,
    size_t inter_threads,
    size_t intra_threads,
    int max_queued_batches)
{
    return std::make_unique<GeneratorWrapper>(
        (std::string)model_path,
        (std::string)device,
        ConvertVector<rust::Vec<int>, std::vector<int>>(std::move(device_indicies)),
        (std::string)compute_type,
        inter_threads,
        intra_threads,
        max_queued_batches);
}