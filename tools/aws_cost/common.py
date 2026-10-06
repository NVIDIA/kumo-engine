# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import requests

# URL base for getting AWS costs is us-east-1 since pricing API doesn't
# have endpoint for every region, but the results include all regions
AWS_COST_ENDPOINT = 'https://pricing.us-east-1.amazonaws.com'


class Ec2EmrPricing:
    def __init__(self, region: str):

        self.region = region

        index_response = requests.get(AWS_COST_ENDPOINT +
                                      '/offers/v1.0/aws/index.json')
        self._pricing_index = index_response.json()

        self._compute_emr_price()
        self._compute_ec2_price()

    def _get_product_pricing(self, product_name: str) -> Dict:
        base_url = self._pricing_index['offers'][product_name][
            'currentRegionIndexUrl']
        regions_response = requests.get(AWS_COST_ENDPOINT + base_url)

        region_result = regions_response.json()['regions'][self.region]
        region_url = AWS_COST_ENDPOINT + region_result['currentVersionUrl']

        return requests.get(region_url).json()

    def _compute_emr_price(self):
        emr_pricing = self._get_product_pricing('ElasticMapReduce')

        sku_to_instance_type = {}
        for sku, sku_val in emr_pricing['products'].items():
            is_emr = sku_val['attributes'].get('softwareType') == 'EMR'
            if is_emr:
                sku_to_instance_type[sku] = \
                    sku_val['attributes']['instanceType']

        self.emr_prices = {}
        for sku in sku_to_instance_type.keys():
            instance_type = sku_to_instance_type.get(sku)
            sku_info = emr_pricing['terms']['OnDemand'][sku]
            if len(sku_info) > 1:
                raise ValueError(f'More than one SKU: {sku_info}')
            _, sku_info_value = sku_info.popitem()
            price_dimensions = sku_info_value['priceDimensions']
            if len(price_dimensions) > 1:
                raise ValueError('More than one price dimension: '
                                 f'{price_dimensions}')
            _, price_dimensions_value = price_dimensions.popitem()
            price = float(price_dimensions_value['pricePerUnit']['USD'])
            self.emr_prices[instance_type] = price

    def _compute_ec2_price(self):
        ec2_pricing = self._get_product_pricing('AmazonEC2')

        ec2_sku_to_instance_type = {}
        for sku in ec2_pricing['products']:
            try:
                attr = ec2_pricing['products'][sku]['attributes']
                if attr['tenancy'] == 'Shared' \
                        and attr['operatingSystem'] == 'Linux'\
                        and attr['operation'] == 'RunInstances' \
                        and attr['capacitystatus'] == 'Used':

                    ec2_sku_to_instance_type[sku] = attr['instanceType']

            except KeyError:
                pass

        self.ec2_prices = {}
        for sku in ec2_sku_to_instance_type.keys():
            instance_type = ec2_sku_to_instance_type.get(sku)
            sku_info = ec2_pricing['terms']['OnDemand'][sku]
            if len(sku_info) > 1:
                raise ValueError(f'More than one SKU: {sku_info}')
            _, sku_info_value = sku_info.popitem()
            price_dimensions = sku_info_value['priceDimensions']
            if len(price_dimensions) > 1:
                raise ValueError('More than one price dimension: '
                                 f'{price_dimensions}')
            _, price_dimensions_value = price_dimensions.popitem()
            price = float(price_dimensions_value['pricePerUnit']['USD'])
            if instance_type in self.ec2_prices:
                raise ValueError(
                    f'Instance price for {instance_type} already added')

            self.ec2_prices[instance_type] = price

    def get_emr_price(self, instance_type) -> float:
        return self.emr_prices[instance_type]

    def get_ec2_price(self, instance_type) -> float:
        return self.ec2_prices[instance_type]
