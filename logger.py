import os

class Logger:
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
   
        self.log_file = log_dir + "/log.txt"
        
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
            
        self.buffer = []        

    def will_write(self, line):
        print(line)
        self.buffers.append(line)

    def flush(self):
        with open(self.log_file, "a") as f:
            f.write("\n".join(self.buffers))
            f.write("\n")
        self.buffers = []

    def write(self, line):
        self.will_write(line)
        self.flush()

    def log_write(self, learn_type, **values):
        """log write in buffers
        ex ) log_write("train", epoch=1, loss=0.3)
        Parmeters:
            learn_type : it must be train, valid or test
            values : values keys in LOG_VALUES
        """
        assert learn_type in ["train", "valid", "test"]
        for k in values.keys():
            if k not in LOG_VALUES[learn_type] and k != LOG_KEYS[learn_type]:
                raise KeyError("%s Log %s keys not in log" % (learn_type, k))

        log = "[%s] %s" % (learn_type, json.dumps(values))
        self.will_write(log)
        if learn_type != "train":
            self.flush()

    def log_parse(self, log_key):
        log_dict = OrderedDict()
        with open(self.log_file, "r") as f:
            for line in f.readlines():
                if len(line) == 1 or not line.startswith("[%s]" % (log_key)):
                    continue
                # line : ~~
                line = line[line.find("] ") + 2:]  # ~~
                line_log = json.loads(line)

                train_log_key = line_log[LOG_KEYS[log_key]]
                line_log.pop(LOG_KEYS[log_key], None)
                log_dict[train_log_key] = line_log
        return log_dict

    def log_plot(self, log_key,
                 figsize=(12, 12), title="plot", colors=["C1", "C2"]):
        fig = plt.figure(figsize=figsize)
        plt.title(title)
        plt.legend(LOG_VALUES[log_key], loc="best")

        ax = plt.subplot(111)
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(LOG_VALUES[log_key])))
        ax.set_prop_cycle(cycler('color', colors))

        log_dict = self.log_parse(log_key)
        x = log_dict.keys()
        for keys in LOG_VALUES[log_key]:
            if keys not in list(log_dict.values())[0]:
                continue
            y = [v[keys] for v in log_dict.values()]

            label = keys + ", max : %f" % (max(y))
            ax.plot(x, y, marker="o", linestyle="solid", label=label)
            if max(y) > 1:
                ax.set_ylim([min(y) - 1, y[0] + 1])
        ax.legend(fontsize=30)

        plt.show()
        
