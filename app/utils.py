import csv


def data_to_csv(device):
    with open('./csv/data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(device.name.split(','))
        writer.writerow(device.values.split(','))