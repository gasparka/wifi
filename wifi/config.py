from dataclasses import dataclass


@dataclass
class Config:
    """
    Table 17-4—Modulation-dependent parameters
    +------------+---------+----------------+-----------------+-----------------+--------------+
    | Modulation |         |                |                 |                 |              |
    |            | Coding  |   Coded bits   |   Coded bits    |    Data bits    |   Data Rate  |
    |            |         |                |                 |                 | (20MHz band) |
    |            |   rate  | per subcarrier | per OFDM symbol | per OFDM symbol |              |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    BPSK    |   1/2   |        1       |        48       |        24       |       6      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    BPSK    |   3/4   |        1       |        48       |        36       |       9      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    QPSK    |   1/2   |        2       |        96       |        48       |      12      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    QPSK    |   3/4   |        2       |        96       |        72       |      18      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   16-QAM   |   1/2   |        4       |       192       |        96       |      24      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   16-QAM   |   3/4   |        4       |       192       |       144       |      36      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   64-QAM   |   2/3   |        6       |       288       |       192       |      48      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   64-QAM   |   3/4   |        6       |       288       |       216       |      54      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    """

    modulation: str
    coding_rate: str
    coded_bits_per_carrier_symbol: int  # i.e. bits per symbol (sps)
    coded_bits_per_ofdm_symbol: int     # i.e. sps * 48, as there are 48 data-carriers
    data_bits_per_ofdm_symbol: int
    data_rate: int

    @classmethod
    def from_data_rate(cls, data_rate: int):
        if data_rate == 6:
            return Config('BPSK', '1/2', 1, 48, 24, 6)
        elif data_rate == 9:
            return Config('BPSK', '3/4', 1, 48, 36, 9)
        elif data_rate == 12:
            return Config('QPSK', '1/2', 2, 96, 48, 12)
        elif data_rate == 18:
            return Config('QPSK', '3/4', 2, 96, 72, 18)
        elif data_rate == 24:
            return Config('16-QAM', '1/2', 4, 192, 96, 24)
        elif data_rate == 36:
            return Config('16-QAM', '3/4', 4, 192, 144, 36)
        elif data_rate == 48:
            return Config('64-QAM', '2/3', 6, 288, 192, 48)
        elif data_rate == 54:
            return Config('64-QAM', '3/4', 6, 288, 216, 54)

