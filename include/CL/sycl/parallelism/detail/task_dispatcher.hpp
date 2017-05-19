#ifndef TRISYCL_SYCL_DETAIL_TASK_DISPATCHER_HPP
#define TRISYCL_SYCL_DETAIL_TASK_DISPATCHER_HPP


#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <queue>
#include <thread>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

class task_dispatcher : public detail::singleton<task_dispatcher> {

private:

  std::vector<std::thread> workers;

  std::queue<std::function<void()>> tasks;

  std::mutex queue_mutex;
  std::condition_variable condition;
  static constexpr size_t pool_size = 4;

  bool stop;

public:

 task_dispatcher(size_t size) : stop { false } {
   for (size_t i = 0; i < size; ++i)
     workers.emplace_back([&] {
         for(;;) {

           std::function<void()> task;
           {
             std::unique_lock<std::mutex> lock(queue_mutex);

             condition.wait(lock, [&]{ return stop || !tasks.empty(); });

             if(stop && tasks.empty()) return;

             task = std::move(tasks.front());
             tasks.pop();
           }
           task();
         }
       });
 }


  task_dispatcher() : task_dispatcher { pool_size } {}


  ~task_dispatcher() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }

    condition.notify_all();
    for(auto& worker: workers)
      worker.join();
  }

  template<class F>
  std::future<void> dispatch(F&& f) {

    auto task = std::make_shared< std::packaged_task<void()> >(f);

    std::future<void> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);

      if(stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      tasks.emplace([=](){ (*task)(); });
    }

    condition.notify_one();
    return res;
  }

};


}
}
}

/*
    # Some Emacs stuff:
    ### Local Variables:
    ### ispell-local-dictionary: "american"
    ### eval: (flyspell-prog-mode)
    ### End:
*/

#endif // TRISYCL_SYCL_DETAIL_TASK_DISPATCHER_HPP
