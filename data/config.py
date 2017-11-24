import ConfigParser

config = ConfigParser.RawConfigParser()


config.add_section('Parameter')
config.set('Parameter', 'batch_size', '128')
config.set('Parameter', 'learning_rate', '0.001')
config.set('Parameter', 'iteration', '1000')
config.set('Parameter', 'hidden_dim', '8')
config.set('Parameter', 'test_ratio', '0.2')
config.set('Parameter', 'train_dir', 'log/train')
config.set('Parameter', 'what_data_use', 'rna')
config.set('Parameter', 'save_result_dir', 'result')


with open('config.cfg', 'wb') as configfile:
	config.write(configfile)
