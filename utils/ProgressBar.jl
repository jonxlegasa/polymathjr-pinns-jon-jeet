module ProgressBar
using ProgressMeter

# ---------------- progress bar ----------------

struct ProgressBarSettings
  maxiters::Int
  message::String
end

function Bar(s::ProgressBarSettings)
  # global_loss_tuple = Tuple{Int64, Float64, Float64, Float64, Float64}[] # this will store the global loss per iteration milestone
  p_bar = Progress(s.maxiters, desc=s.message) # The progress bar.
  # The callback function is called after each optimization step.
  # We use it to update our progress bar.
  global iter_count = 0
  global last_shown_values = 0
  println(last_shown_values)
  callback = function (p, l)
    iter_count += 1
    # If the current iteration is a milestone (1, 100, 200, etc.),
    # update the values that we want to display.
    if iter_count % 100 == 0 || iter_count == 1
      last_shown_values = [(:iter, iter_count), (:loss, l)]
      # Also, ensure the very last iteration's values are shown.
    elseif iter_count == s.maxiters
      last_shown_values = [(:iter, iter_count), (:loss, l)]
    end

    # On every single step, advance the progress bar, but always display
    # the stored "last_shown_values". This makes the text static between milestones.
    ProgressMeter.next!(p_bar; showvalues=last_shown_values)

    return false # Return false to continue the optimization.
  end
  return callback
end

export Bar

end
