import DealCard from './DealCard';

interface DealsGridProps {
  onDealClick: (id: number) => void;
}

export default function DealsGrid({ onDealClick }: DealsGridProps) {
  const deals = [
    { id: 1, name: 'Deal - 1' },
    { id: 2, name: 'Deal - 2' },
    { id: 3, name: 'Deal - 3' },
    { id: 4, name: 'Deal - 4' }
  ];

  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-2 gap-8 max-w-2xl mx-auto">
          {deals.map((deal) => (
            <DealCard
              key={deal.id}
              id={deal.id}
              name={deal.name}
              onClick={onDealClick}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
