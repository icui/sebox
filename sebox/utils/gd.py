if __name__ == '__main__':
    # check misfit values
    from sebox import root

    root.restore()

    for i, optim in enumerate(root):
        if len(optim) < 2 or optim[0].misfit_value is None:
            continue

        print(f'Iteration {i}')
    
        steps = [0.0]
        vals = [optim[0].misfit_value]

        if len(optim):
            for step in optim[-2]:
                steps.append(step.step)
                vals.append(step[1].misfit_value)
        
        for step, val in zip(steps, vals):
            if val is None:
                break
            
            entry = f' {step:.3e}: {val:.3e}'

            if optim.done and val == min(vals):
                entry += ' *'
            
            print(entry)
        
        print()
